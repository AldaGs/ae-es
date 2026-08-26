/*	ExtendedShadow_Kernel_Metal.h

	Metal (MSL) port of ExtendedShadow_Kernel.cu, for the macOS GPU path.

	A faithful 1:1 translation of the CUDA kernels - same per-pixel ray march,
	same bilinear samplers, same bbox crop, same premultiplied composite. The
	host orchestrator lives in ExtendedShadow_Metal.mm and mirrors ES_March_CUDA.

	The MSL source is embedded as a C string and compiled at RUNTIME via
	-[MTLDevice newLibraryWithSource:...], exactly like Bloom and Buildable
	Strokes. Differences from CUDA are pure API, not algorithm:
	  - [[thread_position_in_grid]] instead of blockIdx/threadIdx
	  - ESGpuParams arrives as `constant ESGpuParams&` (set via setBytes)
	  - the device bbox uses MSL atomic_int + atomic_fetch_min/max_explicit
	AE hands worlds as GPU_BGRA128 (float4, .x=B .y=G .z=R .w=A).
*/

#pragma once

static const char *kESKernelMetalString = R"ESMETAL(
#include <metal_stdlib>
using namespace metal;

#define ES_INF_K 1e30f

// LAYOUT MUST MATCH ESGpuParams in ExtendedShadow.cpp / _Metal.mm / _Kernel.cu.
struct ESGpuParams {
    int   type;
    float ddx, ddy;
    float length;
    float srcX, srcY;
    float lengthPct;
    float fadeIn, fadeOut;
    float tint, opacity, threshold;
    int   fill;
    float col[3];
    float col2[3];
    float gsx, gsy, gex, gey;
    int   offX, offY;
    int   oox, ooy;
    int   inW, inH;
    int   W, H;
    int   srcPitch, dstPitch;
    float step;
    int   pad;
    int   wx0, wy0, wx1, wy1;
};

/* ---- device samplers (bilinear, zero outside the input world) ----------- */

inline float4 es_tap4(const device float4 *s, int pitch, int inW, int inH, int x, int y) {
    if (x < 0 || y < 0 || x >= inW || y >= inH) return float4(0.0f);
    return s[(uint)y * pitch + x];
}

inline float es_sampA(const device float4 *s, int pitch, int inW, int inH, float fu, float fv) {
    float fx = floor(fu), fy = floor(fv);
    int x0 = (int)fx, y0 = (int)fy;
    float tx = fu - fx, ty = fv - fy;
    float a = es_tap4(s, pitch, inW, inH, x0,     y0    ).w;
    float b = es_tap4(s, pitch, inW, inH, x0 + 1, y0    ).w;
    float c = es_tap4(s, pitch, inW, inH, x0,     y0 + 1).w;
    float d = es_tap4(s, pitch, inW, inH, x0 + 1, y0 + 1).w;
    float top = a + (b - a) * tx, bot = c + (d - c) * tx;
    return top + (bot - top) * ty;
}

inline void es_sampRGB(const device float4 *s, int pitch, int inW, int inH, float fu, float fv,
                       thread float *oR, thread float *oG, thread float *oB) {
    float fx = floor(fu), fy = floor(fv);
    int x0 = (int)fx, y0 = (int)fy;
    float tx = fu - fx, ty = fv - fy;
    float4 a = es_tap4(s, pitch, inW, inH, x0,     y0    );
    float4 b = es_tap4(s, pitch, inW, inH, x0 + 1, y0    );
    float4 c = es_tap4(s, pitch, inW, inH, x0,     y0 + 1);
    float4 d = es_tap4(s, pitch, inW, inH, x0 + 1, y0 + 1);
    float rt = a.z + (b.z - a.z) * tx, rb = c.z + (d.z - c.z) * tx; *oR = rt + (rb - rt) * ty;
    float gt = a.y + (b.y - a.y) * tx, gb = c.y + (d.y - c.y) * tx; *oG = gt + (gb - gt) * ty;
    float bt = a.x + (b.x - a.x) * tx, bb = c.x + (d.x - c.x) * tx; *oB = bt + (bb - bt) * ty;
}

/* ---- shape bbox (atomic reduction over the input alpha) ----------------- */

kernel void es_bbox(const device float4 *src [[buffer(0)]],
                    device atomic_int   *box [[buffer(1)]],
                    constant ESGpuParams &p  [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    int u = gid.x, v = gid.y;
    if (u >= p.inW || v >= p.inH) return;
    if (src[(uint)v * p.srcPitch + u].w >= p.threshold) {
        atomic_fetch_min_explicit(&box[0], u, memory_order_relaxed);
        atomic_fetch_min_explicit(&box[1], v, memory_order_relaxed);
        atomic_fetch_max_explicit(&box[2], u, memory_order_relaxed);
        atomic_fetch_max_explicit(&box[3], v, memory_order_relaxed);
    }
}

/* ---- the march kernel --------------------------------------------------- */

kernel void es_march(const device float4 *src [[buffer(0)]],
                     device float4       *dst [[buffer(1)]],
                     constant ESGpuParams &p  [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= p.W || y >= p.H) return;

    float4 art = es_tap4(src, p.srcPitch, p.inW, p.inH, x - p.offX, y - p.offY);

    // Outside the work rect no shadow can reach: copy the source through.
    if (x < p.wx0 || x > p.wx1 || y < p.wy0 || y > p.wy1) {
        dst[(uint)y * p.dstPitch + x] = art;
        return;
    }

    float iu0 = (float)(x - p.offX);
    float iv0 = (float)(y - p.offY);

    float Lx = (float)(x + p.oox), Ly = (float)(y + p.ooy);
    float dx, dy, L;
    if (p.type == 1) {			// directional
        dx = p.ddx; dy = p.ddy; L = p.length;
    } else {
        float vx = Lx - p.srcX, vy = Ly - p.srcY;
        float dist = sqrt(vx * vx + vy * vy);
        if (dist < 1e-3f) { dx = dy = 0.0f; L = 0.0f; }
        else {
            float inv = 1.0f / dist;
            if (p.type == 3) { dx = -vx * inv; dy = -vy * inv; }	// inverse
            else             { dx =  vx * inv; dy =  vy * inv; }	// radial
            L = p.lengthPct * dist;
            if (L > p.length) L = p.length;
        }
    }

    // clip the march to where the ray leaves the input world
    float tMax = L;
    if (dx >  1e-6f) { float tb = iu0 / dx;                     if (tb < tMax) tMax = tb; }
    else if (dx < -1e-6f) { float tb = (iu0 - (p.inW - 1)) / dx; if (tb < tMax) tMax = tb; }
    if (dy >  1e-6f) { float tb = iv0 / dy;                     if (tb < tMax) tMax = tb; }
    else if (dy < -1e-6f) { float tb = (iv0 - (p.inH - 1)) / dy; if (tb < tMax) tMax = tb; }
    if (tMax < 0.0f) tMax = 0.0f;

    const float thr = p.threshold;
    float cov = es_sampA(src, p.srcPitch, p.inW, p.inH, iu0, iv0);
    float fadeDist = (cov >= thr) ? 0.0f : ES_INF_K;
    float hitU = iu0, hitV = iv0;

    for (float t = p.step; t <= tMax + 1e-4f; t += p.step) {
        float su = iu0 - dx * t, sv = iv0 - dy * t;
        float a = es_sampA(src, p.srcPitch, p.inW, p.inH, su, sv);
        if (a > cov) cov = a;
        if (fadeDist == ES_INF_K && a >= thr) { fadeDist = t; hitU = su; hitV = sv; }
    }

    float4 outPix;
    if (cov <= 0.0f) {
        outPix = art;
    } else {
        if (fadeDist == ES_INF_K) fadeDist = (L > 0.0f) ? L : 0.0f;
        float s = (L > 1e-6f) ? (fadeDist / L) : 0.0f;
        s = clamp(s, 0.0f, 1.0f);
        float ramp = p.fadeIn + (p.fadeOut - p.fadeIn) * s;

        float fr = p.col[0], fg = p.col[1], fb = p.col[2];
        if (p.fill == 2 || p.fill == 3) {			// 2=linear, 3=radial
            float gdx = p.gex - p.gsx, gdy = p.gey - p.gsy;
            float gden = gdx * gdx + gdy * gdy;
            float rx = Lx - p.gsx, ry = Ly - p.gsy;
            float tg;
            if (p.fill == 3) {
                float rinv = (gden > 1e-6f) ? 1.0f / sqrt(gden) : 0.0f;
                tg = sqrt(rx * rx + ry * ry) * rinv;
            } else {
                float ginv = (gden > 1e-6f) ? 1.0f / gden : 0.0f;
                tg = (rx * gdx + ry * gdy) * ginv;
            }
            tg = clamp(tg, 0.0f, 1.0f);
            fr = p.col[0] + (p.col2[0] - p.col[0]) * tg;
            fg = p.col[1] + (p.col2[1] - p.col[1]) * tg;
            fb = p.col[2] + (p.col2[2] - p.col[2]) * tg;
        }

        float shR = fr, shG = fg, shB = fb;
        if (p.tint < 1.0f) {
            float oR, oG, oB;
            es_sampRGB(src, p.srcPitch, p.inW, p.inH, hitU, hitV, &oR, &oG, &oB);
            shR = oR * (1.0f - p.tint) + fr * p.tint;
            shG = oG * (1.0f - p.tint) + fg * p.tint;
            shB = oB * (1.0f - p.tint) + fb * p.tint;
        }

        float shA = cov * ramp * p.opacity;
        shA = clamp(shA, 0.0f, 1.0f);

        float aR = art.z, aG = art.y, aB = art.x, aA = art.w;
        float outA = aA + shA * (1.0f - aA);
        float outR, outG, outB;
        if (outA > 1e-6f) {
            float inv = 1.0f / outA;
            outR = (aR * aA + shR * shA * (1.0f - aA)) * inv;
            outG = (aG * aA + shG * shA * (1.0f - aA)) * inv;
            outB = (aB * aA + shB * shA * (1.0f - aA)) * inv;
        } else { outR = outG = outB = 0.0f; }
        outPix = float4(outB, outG, outR, outA);	// back to BGRA
    }

    dst[(uint)y * p.dstPitch + x] = outPix;
}
)ESMETAL";
