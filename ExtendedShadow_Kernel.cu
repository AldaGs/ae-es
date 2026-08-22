/*	ExtendedShadow_Kernel.cu

	GPU (CUDA) path for Extended Shadow.

	STAGE 2: the real MARCH, on the GPU. One thread = one output pixel, running
	the SAME max-along-ray loop the CPU SmartRender does. Because the march is
	per-pixel independent, this is a near-direct translation - no JFA / algorithm
	redesign (unlike Buildable Stroke's distance transforms).

	Differences from the CPU path (intentional for this stage):
	  - NO bbox crop. We march over the full output W x H and sample the INPUT
	    world directly from device memory (bilinear), rather than a cropped
	    alphaBuf. Simpler; the crop is a perf optimization for a later stage.
	  - Sampling outside the input world returns 0 (transparent), matching the
	    CPU's zero-outside behaviour at the buffer edge.

	AE GPU worlds are GPU_BGRA128 = float4 per pixel, order B,G,R,A (so alpha is
	.w, and R=.z, G=.y, B=.x). Row pitch is in ELEMENTS (rowbytes / 16).

	The host fills ESGpuParams (identical layout to the copy in ExtendedShadow.cpp)
	and launches one thread per output pixel.
*/

#include <cuda_runtime.h>

#define ES_INF_K	1e30f

struct ESGpuParams {
	int		type;			// 1 dir, 2 radial, 3 inverse-radial
	float	ddx, ddy;		// directional projection dir (precomputed on host)
	float	length;			// px (reach / cap)
	float	srcX, srcY;		// radial source, layer px
	float	lengthPct;		// fraction (radial)
	float	fadeIn, fadeOut;// 0..1
	float	tint, opacity, threshold;	// 0..1
	int		fill;			// 1 solid, 2 linear gradient
	float	col[3];			// solid / gradient start RGB
	float	col2[3];		// gradient end RGB
	float	gsx, gsy, gex, gey;	// gradient endpoints, layer px
	int		offX, offY;		// input->output offset (inRect.left-outRect.left, ...)
	int		oox, ooy;		// output->layer origin (outRect.left/top)
	int		inW, inH;		// input world size
	int		W, H;			// output world size
	int		srcPitch, dstPitch;
	float	step;			// march sample spacing, px
	int		pad;			// shadow reach in px (== max L); grows the work rect
	int		wx0, wy0, wx1, wy1;	// work rect in OUTPUT space (host fills after bbox)
};

/* ---- device samplers (bilinear, zero outside the input world) ----------- */

__device__ __forceinline__ float4
es_tap4 (const float4 *s, int pitch, int inW, int inH, int x, int y)
{
	if (x < 0 || y < 0 || x >= inW || y >= inH) return make_float4(0.f, 0.f, 0.f, 0.f);
	return s[(size_t)y * pitch + x];
}

__device__ __forceinline__ float
es_sampA (const float4 *s, int pitch, int inW, int inH, float fu, float fv)
{
	float fx = floorf(fu), fy = floorf(fv);
	int x0 = (int)fx, y0 = (int)fy;
	float tx = fu - fx, ty = fv - fy;
	float a = es_tap4(s, pitch, inW, inH, x0,     y0    ).w;
	float b = es_tap4(s, pitch, inW, inH, x0 + 1, y0    ).w;
	float c = es_tap4(s, pitch, inW, inH, x0,     y0 + 1).w;
	float d = es_tap4(s, pitch, inW, inH, x0 + 1, y0 + 1).w;
	float top = a + (b - a) * tx, bot = c + (d - c) * tx;
	return top + (bot - top) * ty;
}

__device__ __forceinline__ void
es_sampRGB (const float4 *s, int pitch, int inW, int inH, float fu, float fv,
			float *oR, float *oG, float *oB)
{
	float fx = floorf(fu), fy = floorf(fv);
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

__global__ void
es_bbox (const float4 *src, int pitch, int inW, int inH, float thr, int *box)
{
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= inW || v >= inH) return;
	if (src[(size_t)v * pitch + u].w >= thr) {
		atomicMin(&box[0], u);  atomicMin(&box[1], v);
		atomicMax(&box[2], u);  atomicMax(&box[3], v);
	}
}

/* ---- the march kernel --------------------------------------------------- */

__global__ void
es_march (const float4 *src, float4 *dst, ESGpuParams p)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= p.W || y >= p.H) return;

	// source art at this output pixel (input space = output - offset)
	float4 art = es_tap4(src, p.srcPitch, p.inW, p.inH, x - p.offX, y - p.offY);

	// Outside the work rect (shape bbox + pad) no shadow can reach: the march
	// would only ever sample alpha 0. Copy the source through and skip it - this
	// is the whole point of the crop, since most of a frame is empty.
	if (x < p.wx0 || x > p.wx1 || y < p.wy0 || y > p.wy1) {
		dst[(size_t)y * p.dstPitch + x] = art;
		return;
	}

	float iu0 = (float)(x - p.offX);
	float iv0 = (float)(y - p.offY);

	// projection direction + length for this pixel
	float Lx = (float)(x + p.oox), Ly = (float)(y + p.ooy);
	float dx, dy, L;
	if (p.type == 1) {			// directional
		dx = p.ddx; dy = p.ddy; L = p.length;
	} else {
		float vx = Lx - p.srcX, vy = Ly - p.srcY;
		float dist = sqrtf(vx * vx + vy * vy);
		if (dist < 1e-3f) { dx = dy = 0.f; L = 0.f; }
		else {
			float inv = 1.f / dist;
			if (p.type == 3) { dx = -vx * inv; dy = -vy * inv; }	// inverse
			else             { dx =  vx * inv; dy =  vy * inv; }	// radial
			L = p.lengthPct * dist;
			if (L > p.length) L = p.length;		// capped by Length
		}
	}

	// clip the march to where the ray leaves the input world
	float tMax = L;
	if (dx >  1e-6f) { float tb = iu0 / dx;                 if (tb < tMax) tMax = tb; }
	else if (dx < -1e-6f) { float tb = (iu0 - (p.inW - 1)) / dx; if (tb < tMax) tMax = tb; }
	if (dy >  1e-6f) { float tb = iv0 / dy;                 if (tb < tMax) tMax = tb; }
	else if (dy < -1e-6f) { float tb = (iv0 - (p.inH - 1)) / dy; if (tb < tMax) tMax = tb; }
	if (tMax < 0.f) tMax = 0.f;

	const float thr = p.threshold;
	float cov = es_sampA(src, p.srcPitch, p.inW, p.inH, iu0, iv0);
	float fadeDist = (cov >= thr) ? 0.f : ES_INF_K;
	float hitU = iu0, hitV = iv0;

	for (float t = p.step; t <= tMax + 1e-4f; t += p.step) {
		float su = iu0 - dx * t, sv = iv0 - dy * t;
		float a = es_sampA(src, p.srcPitch, p.inW, p.inH, su, sv);
		if (a > cov) cov = a;
		if (fadeDist == ES_INF_K && a >= thr) { fadeDist = t; hitU = su; hitV = sv; }
	}

	float4 outPix;
	if (cov <= 0.f) {
		outPix = art;			// nothing to shadow here
	} else {
		if (fadeDist == ES_INF_K) fadeDist = (L > 0.f) ? L : 0.f;
		float s = (L > 1e-6f) ? (fadeDist / L) : 0.f;
		if (s < 0.f) s = 0.f; else if (s > 1.f) s = 1.f;
		float ramp = p.fadeIn + (p.fadeOut - p.fadeIn) * s;

		// fill colour: solid, or lerp along the gradient axis (layer space)
		float fr = p.col[0], fg = p.col[1], fb = p.col[2];
		if (p.fill == 2) {
			float gdx = p.gex - p.gsx, gdy = p.gey - p.gsy;
			float gden = gdx * gdx + gdy * gdy;
			float ginv = (gden > 1e-6f) ? 1.f / gden : 0.f;
			float tg = ((Lx - p.gsx) * gdx + (Ly - p.gsy) * gdy) * ginv;
			if (tg < 0.f) tg = 0.f; else if (tg > 1.f) tg = 1.f;
			fr = p.col[0] + (p.col2[0] - p.col[0]) * tg;
			fg = p.col[1] + (p.col2[1] - p.col[1]) * tg;
			fb = p.col[2] + (p.col2[2] - p.col[2]) * tg;
		}

		// tint toward shadow hue (tint<1 keeps the object's colour at the hit)
		float shR = fr, shG = fg, shB = fb;
		if (p.tint < 1.f) {
			float oR, oG, oB;
			es_sampRGB(src, p.srcPitch, p.inW, p.inH, hitU, hitV, &oR, &oG, &oB);
			shR = oR * (1.f - p.tint) + fr * p.tint;
			shG = oG * (1.f - p.tint) + fg * p.tint;
			shB = oB * (1.f - p.tint) + fb * p.tint;
		}

		float shA = cov * ramp * p.opacity;
		if (shA < 0.f) shA = 0.f; else if (shA > 1.f) shA = 1.f;

		// composite: source art OVER its shadow (straight alpha). art is BGRA.
		float aR = art.z, aG = art.y, aB = art.x, aA = art.w;
		float outA = aA + shA * (1.f - aA);
		float outR, outG, outB;
		if (outA > 1e-6f) {
			float inv = 1.f / outA;
			outR = (aR * aA + shR * shA * (1.f - aA)) * inv;
			outG = (aG * aA + shG * shA * (1.f - aA)) * inv;
			outB = (aB * aA + shB * shA * (1.f - aA)) * inv;
		} else { outR = outG = outB = 0.f; }
		outPix = make_float4(outB, outG, outR, outA);	// back to BGRA
	}

	dst[(size_t)y * p.dstPitch + x] = outPix;
}

// No extern "C": nvcc uses the same MSVC host compiler (-ccbin), so the mangled
// name matches the extern declaration in the .cpp.
void
ES_March_CUDA (const float *src, float *dst, const ESGpuParams &pin)
{
	ESGpuParams p = pin;

	// --- pass 1: find the shape bbox on-device, then crop the march to it ---
	// Default to the whole frame; if the bbox pass succeeds we tighten it.
	p.wx0 = 0; p.wy0 = 0; p.wx1 = p.W - 1; p.wy1 = p.H - 1;

	int *dbox = NULL;
	if (cudaMalloc(&dbox, 4 * sizeof(int)) == cudaSuccess && dbox) {
		int hbox[4] = { p.inW, p.inH, -1, -1 };		// min inits high, max inits low
		cudaMemcpy(dbox, hbox, 4 * sizeof(int), cudaMemcpyHostToDevice);

		dim3 bblock(16, 16);
		dim3 bgrid((p.inW + bblock.x - 1) / bblock.x, (p.inH + bblock.y - 1) / bblock.y);
		es_bbox<<<bgrid, bblock>>>((const float4*)src, p.srcPitch, p.inW, p.inH,
								   p.threshold, dbox);

		cudaMemcpy(hbox, dbox, 4 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(dbox);

		if (hbox[2] < 0) {
			// No shape anywhere -> empty work rect so every pixel copies through.
			p.wx0 = 1; p.wy0 = 1; p.wx1 = 0; p.wy1 = 0;
		} else {
			// Shape bbox is in INPUT coords; output = input + offset, then +/- pad.
			int wx0 = hbox[0] + p.offX - p.pad;
			int wy0 = hbox[1] + p.offY - p.pad;
			int wx1 = hbox[2] + p.offX + p.pad;
			int wy1 = hbox[3] + p.offY + p.pad;
			if (wx0 < 0) wx0 = 0;  if (wy0 < 0) wy0 = 0;
			if (wx1 > p.W - 1) wx1 = p.W - 1;
			if (wy1 > p.H - 1) wy1 = p.H - 1;
			p.wx0 = wx0; p.wy0 = wy0; p.wx1 = wx1; p.wy1 = wy1;
		}
	}

	// --- pass 2: march (empty pixels early-out to a copy inside the kernel) --
	dim3 block(16, 16);
	dim3 grid((p.W + block.x - 1) / block.x, (p.H + block.y - 1) / block.y);
	es_march<<<grid, block>>>((const float4*)src, (float4*)dst, p);
}
