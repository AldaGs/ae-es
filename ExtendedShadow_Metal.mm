/*	ExtendedShadow_Metal.mm

	macOS Metal host for the ExtendedShadow GPU path. Three entry points, mirrored
	on Bloom_Metal.mm / BuildableStroke_Metal.mm:

	  ES_MetalCompile  - GPU_DEVICE_SETUP: compile the MSL library once per device
	                     and build one pipeline state per kernel; stash in gpu_data.
	  ES_MetalDispose  - GPU_DEVICE_SETDOWN: release the pipeline states.
	  ES_March_Metal   - per frame: a step-for-step twin of ES_March_CUDA. Device
	                     bbox (own command buffer, read back) -> crop -> one march
	                     over the full output world.

	The MSL source lives in ExtendedShadow_Kernel_Metal.h and is compiled at
	runtime via -[MTLDevice newLibraryWithSource:...].
*/

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>

#include "ExtendedShadow_Kernel_Metal.h"

/* LAYOUT MUST MATCH ESGpuParams in ExtendedShadow_Kernel_Metal.h and the copy in
   ExtendedShadow.cpp exactly. */
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

struct ESMetalGPUData {
	id<MTLComputePipelineState> bbox, march;
};

/* ------------------------------------------------------- compile / dispose */

extern "C" void ES_MetalDispose (void *dataPV);	// defined below; used in Compile

extern "C" bool
ES_MetalCompile (void *devicePV, void **outData, char *errBuf, int errLen)
{
	@autoreleasepool {
		id<MTLDevice> device = (id<MTLDevice>)devicePV;
		NSError *error = nil;
		NSString *source = [NSString stringWithUTF8String:kESKernelMetalString];
		id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
		if (!lib) {
			if (errBuf && errLen > 0) {
				const char *m = error ? [[error localizedDescription] UTF8String]
									   : "unknown Metal compile error";
				strncpy(errBuf, m ? m : "nil", errLen - 1);
				errBuf[errLen - 1] = 0;
			}
			return false;
		}

		ESMetalGPUData *d = (ESMetalGPUData*)calloc(1, sizeof(ESMetalGPUData));
		bool ok = true;

		auto mk = [&](id<MTLComputePipelineState> __strong *slot, const char *name) {
			if (!ok) return;
			id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
			if (!fn) {
				if (errBuf && errLen > 0) { snprintf(errBuf, errLen, "missing kernel: %s", name); }
				ok = false; return;
			}
			NSError *e = nil;
			*slot = [device newComputePipelineStateWithFunction:fn error:&e];
			[fn release];
			if (!*slot) {
				if (errBuf && errLen > 0) {
					const char *m = e ? [[e localizedDescription] UTF8String] : "pso failed";
					snprintf(errBuf, errLen, "pso %s: %s", name, m ? m : "nil");
				}
				ok = false;
			}
		};

		mk(&d->bbox,  "es_bbox");
		mk(&d->march, "es_march");

		[lib release];

		if (!ok) { ES_MetalDispose(d); return false; }
		*outData = d;
		return true;
	}
}

extern "C" void
ES_MetalDispose (void *dataPV)
{
	ESMetalGPUData *d = (ESMetalGPUData*)dataPV;
	if (!d) return;
	[d->bbox release];
	[d->march release];
	free(d);
}

/* -------------------------------------------------------------- the render */

extern "C" bool
ES_March_Metal (void *devicePV, void *queuePV, void *dataPV,
				void *srcMemPV, void *dstMemPV, ESGpuParams p)
{
	@autoreleasepool {
		id<MTLDevice>       device = (id<MTLDevice>)devicePV;
		id<MTLCommandQueue> queue  = (id<MTLCommandQueue>)queuePV;
		ESMetalGPUData     *d      = (ESMetalGPUData*)dataPV;
		id<MTLBuffer>       src    = (id<MTLBuffer>)srcMemPV;
		id<MTLBuffer>       dst    = (id<MTLBuffer>)dstMemPV;

		const MTLSize TG = MTLSizeMake(16, 16, 1);
		auto disp2D = [&](id<MTLComputeCommandEncoder> e, int w, int h) {
			[e dispatchThreadgroups:MTLSizeMake((w+15)/16, (h+15)/16, 1)
				threadsPerThreadgroup:TG];
		};

		// Default to the whole frame; tighten it if the bbox pass succeeds.
		p.wx0 = 0; p.wy0 = 0; p.wx1 = p.W - 1; p.wy1 = p.H - 1;

		// --- pass 1: shape bbox on-device, read back to compute the crop -----
		int hbox[4] = { p.inW, p.inH, -1, -1 };		// min inits high, max inits low
		id<MTLBuffer> boxBuf = [device newBufferWithLength:4*sizeof(int)
											options:MTLResourceStorageModeShared];
		memcpy([boxBuf contents], hbox, sizeof(hbox));
		{
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:d->bbox];
			[e setBuffer:src    offset:0 atIndex:0];
			[e setBuffer:boxBuf offset:0 atIndex:1];
			[e setBytes:&p length:sizeof(p) atIndex:2];
			disp2D(e, p.inW, p.inH);
			[e endEncoding];
			[cb commit];
			[cb waitUntilCompleted];
			memcpy(hbox, [boxBuf contents], sizeof(hbox));
		}
		[boxBuf release];

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

		// --- pass 2: march (empty pixels early-out to a copy inside the kernel) -
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:d->march];
		[e setBuffer:src offset:0 atIndex:0];
		[e setBuffer:dst offset:0 atIndex:1];
		[e setBytes:&p length:sizeof(p) atIndex:2];
		disp2D(e, p.W, p.H);
		[e endEncoding];
		[cb commit];
		[cb waitUntilCompleted];

		return (cb.status != MTLCommandBufferStatusError);
	}
}
