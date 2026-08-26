/*	ExtendedShadow.cpp

	Extended Shadow - fourth learning plugin. A long shadow: the layer's alpha
	silhouette projected along a direction (or radially from a point), filled
	with a colour / gradient, faded along its length.

	Ported from python-proto/long_shadow/ls_step1..3. Like Buildable Stroke it
	is a SmartFX buffer-expanding effect (the shadow lives outside the layer),
	but the engine is a MARCH, not a distance transform:

	    cov(p) = max over t in [0, L] of  alpha_src(p - t * d)     (bilinear)

	Reading the source's SOFT alpha (not a thresholded mask) makes the shadow
	anti-aliased by construction - no supersampling, unlike the plugins that
	project a binary silhouette and blur afterwards.

	STAGE 1: CPU only, single shadow, the march for every shadow type. The
	width-independent log-doubling fast path (directional) and the CUDA port are
	later stages, exactly as Buildable Stroke was staged.

	Revision History
	Version		Change											Engineer	Date
	=======		======											========	======
	1.0			Initial scaffold from Python prototype		aldai		8/18/2026
*/

#define _CRT_SECURE_NO_WARNINGS

#if HAS_CUDA
	#include <cuda_runtime.h>
	// cuda_runtime.h defines these; our header needs its own versions.
	#undef MAJOR_VERSION
	#undef MINOR_VERSION
#endif

#include "ExtendedShadow.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <thread>
#include <atomic>

/* ---- GPU rendering -------------------------------------------------------
   Flip ES_GPU_RENDER to 1 once the real march kernel exists. At 0 the GPU path
   is COMPILED and LINKED and AE's GPU_DEVICE_SETUP runs (proving toolchain +
   device plumbing), but PreRender does NOT opt this frame into GPU, so every
   frame still renders on the CPU and output stays correct while we build. */
#define ES_GPU_RENDER 1		// STAGE 2: real march kernel

#if HAS_CUDA || HAS_METAL
// POD mirror of the kernel's ESGpuParams - the LAYOUT MUST MATCH the struct in
// ExtendedShadow_Kernel.cu / ExtendedShadow_Kernel_Metal.h exactly (same fields,
// same order). Kept here as a plain struct so the .cpp needn't pull CUDA/Metal
// headers.
struct ESGpuParams {
	int		type;
	float	ddx, ddy;
	float	length;
	float	srcX, srcY;
	float	lengthPct;
	float	fadeIn, fadeOut;
	float	tint, opacity, threshold;
	int		fill;
	float	col[3];
	float	col2[3];
	float	gsx, gsy, gex, gey;
	int		offX, offY;
	int		oox, ooy;
	int		inW, inH;
	int		W, H;
	int		srcPitch, dstPitch;
	float	step;
	int		pad;
	int		wx0, wy0, wx1, wy1;
};

#if HAS_CUDA
// Defined in ExtendedShadow_Kernel.cu. No extern "C": nvcc uses the same MSVC
// host compiler (-ccbin), so the mangled names match.
extern void ES_March_CUDA(const float *src, float *dst, const ESGpuParams &p);
#endif

#if HAS_METAL
// Defined in ExtendedShadow_Metal.mm (Objective-C++). extern "C" so the .cpp
// links against unmangled names.
extern "C" bool ES_MetalCompile (void *devicePV, void **outData, char *errBuf, int errLen);
extern "C" void ES_MetalDispose (void *dataPV);
extern "C" bool ES_March_Metal (void *devicePV, void *queuePV, void *dataPV,
								void *srcMemPV, void *dstMemPV, ESGpuParams p);
#endif
#endif

#define ES_PI		3.14159265358979323846
#define ES_INF		1e30f

/* Set once in GlobalSetup. Needed to reach the AEGP stream suites, which are the
   ONLY way to change param visibility at runtime in AE (learned on BS). */
static AEGP_PluginID	S_es_id = 0L;

/* =========================================================================
   Boilerplate
   ========================================================================= */

static PF_Err
About (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	suites.ANSICallbacksSuite1()->sprintf(
		out_data->return_msg,
		"%s v%d.%d\r%s",
		STR(StrID_Name), MAJOR_VERSION, MINOR_VERSION, STR(StrID_Description));
	return PF_Err_NONE;
}

static PF_Err
GlobalSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	out_data->my_version = PF_VERSION(	MAJOR_VERSION, MINOR_VERSION,
										BUG_VERSION, STAGE_VERSION, BUILD_VERSION);

	// Needed for the DynamicStream suite used to show/hide type- and fill-
	// specific params at runtime.
	suites.UtilitySuite3()->AEGP_RegisterWithAEGP(NULL, STR(StrID_Name), &S_es_id);

	// I_EXPAND_BUFFER: the shadow extends beyond the layer bounds.
	// SEND_UPDATE_PARAMS_UI: needed to sync param visibility on load.
	out_data->out_flags  =	PF_OutFlag_DEEP_COLOR_AWARE |
							PF_OutFlag_I_EXPAND_BUFFER |
							PF_OutFlag_SEND_UPDATE_PARAMS_UI;

	// SmartFX + float + MFR. MUST match PiPL OutFlags_2 exactly (0x0A001400).
	out_data->out_flags2 =	PF_OutFlag2_SUPPORTS_SMART_RENDER |
							PF_OutFlag2_FLOAT_COLOR_AWARE |
							PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

	// Advertise CUDA GPU support (F32 only). Premiere negotiates GPU differently,
	// so gate it to AE. This alone makes AE call GPU_DEVICE_SETUP; a frame only
	// renders on GPU if PreRender ALSO opts in (see ES_GPU_RENDER).
	if (in_data->appl_id != 'PrMr') {
		out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}

	return PF_Err_NONE;
}

/* =========================================================================
   Parameters. A single flat list (no dynamic groups - there is exactly one
   shadow). Gradient controls stay visible even for a Solid fill in this stage;
   the show/hide-when-Solid polish is a later addition (it needs the AEGP stream
   suites, as learned in Buildable Stroke).
   ========================================================================= */

static PF_Err
ParamsSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err		err = PF_Err_NONE;
	PF_ParamDef	def;

	// SUPERVISE so changing it fires PF_Cmd_USER_CHANGED_PARAM, where we show/hide
	// the directional-vs-radial controls. PF_ADD_POPUP does not clear def, so
	// setting def.flags first sticks (confirmed on BS).
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(	STR(StrID_Type_Param_Name),
					3, ES_TYPE_DIR, STR(StrID_Type_Choices), TYPE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_ANGLE(	STR(StrID_Direction_Param_Name), 180, DIRECTION_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Length_Param_Name),
							0, ES_LENGTH_MAX, 0, 300, ES_LENGTH_DFLT,
							PF_Precision_TENTHS, 0, 0, LENGTH_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT(	STR(StrID_Source_Param_Name), 50, 50, 0, SOURCE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_LengthPct_Param_Name),
							0, 200, 0, 100, 30,
							PF_Precision_TENTHS, 0, 0, LENGTH_PCT_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_FadeIn_Param_Name),
							0, 100, 0, 100, 100,
							PF_Precision_TENTHS, 0, 0, FADE_IN_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_FadeOut_Param_Name),
							0, 100, 0, 100, 100,
							PF_Precision_TENTHS, 0, 0, FADE_OUT_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(	STR(StrID_Color_Param_Name), 0, 0, 0, COLOR_DISK_ID);

	// SUPERVISE so toggling Fill fires PF_Cmd_USER_CHANGED_PARAM, where we
	// show/hide the gradient-only controls.
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(	STR(StrID_Fill_Param_Name),
					3, ES_FILL_SOLID, STR(StrID_Fill_Choices), FILL_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(	STR(StrID_Color2_Param_Name), 128, 128, 128, COLOR2_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT(	STR(StrID_GStart_Param_Name), 25, 50, 0, GSTART_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT(	STR(StrID_GEnd_Param_Name), 75, 50, 0, GEND_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Tint_Param_Name),
							0, 100, 0, 100, 100,
							PF_Precision_TENTHS, 0, 0, TINT_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Opacity_Param_Name),
							0, 100, 0, 100, 100,
							PF_Precision_TENTHS, 0, 0, OPACITY_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Threshold_Param_Name),
							0, 100, 0, 100, 50,
							PF_Precision_TENTHS, 0, 0, THRESHOLD_DISK_ID);

	out_data->num_params = ES_NUM_PARAMS;
	return err;
}

/* =========================================================================
   Dynamic show/hide of the type- and fill-specific controls.

   Same mechanism as Buildable Stroke: PF_PUI_INVISIBLE is honored only when a
   param is CREATED, so runtime visibility must go through the AEGP DynamicStream
   suite. Hiding a param's stream hides that control. Premiere has no stream
   suites, so we bail there (it just shows everything).
   ========================================================================= */

static PF_Err
ES_SetParamVisible (
	PF_InData	*in_data,
	A_long		index,
	bool		visible )
{
	PF_Err				err = PF_Err_NONE, err2 = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);
	AEGP_EffectRefH		meH		= NULL;
	AEGP_StreamRefH		streamH	= NULL;

	ERR(suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(
			S_es_id, in_data->effect_ref, &meH));
	ERR(suites.StreamSuite2()->AEGP_GetNewEffectStreamByIndex(
			S_es_id, meH, index, &streamH));
	if (!err && streamH) {
		ERR(suites.DynamicStreamSuite2()->AEGP_SetDynamicStreamFlag(
				streamH, AEGP_DynStreamFlag_HIDDEN, FALSE, visible ? FALSE : TRUE));
		ERR2(suites.StreamSuite2()->AEGP_DisposeStream(streamH));
	}
	if (meH) ERR2(suites.EffectSuite2()->AEGP_DisposeEffect(meH));
	return err;
}

/* Make the UI match the current Shadow Type + Fill. Called on load
   (UPDATE_PARAMS_UI) and whenever either popup changes (USER_CHANGED_PARAM). */
static PF_Err
ES_SyncUI (
	PF_InData	*in_data,
	PF_ParamDef	*params[] )
{
	PF_Err	err = PF_Err_NONE;
	if (in_data->appl_id == 'PrMr') return PF_Err_NONE;

	bool isDir  = (params[ES_TYPE]->u.pd.value == ES_TYPE_DIR);
	bool isGrad = (params[ES_FILL]->u.pd.value != ES_FILL_SOLID);

	// Directional: Shadow Direction only. Radial / Inverse: Source + Length %.
	// (Shadow Length applies to BOTH - it caps radial reach - so it stays shown.)
	ERR(ES_SetParamVisible(in_data, ES_DIRECTION,  isDir));
	ERR(ES_SetParamVisible(in_data, ES_SOURCE,     !isDir));
	ERR(ES_SetParamVisible(in_data, ES_LENGTH_PCT, !isDir));

	// Gradient controls only for a Linear Gradient fill.
	ERR(ES_SetParamVisible(in_data, ES_COLOR2, isGrad));
	ERR(ES_SetParamVisible(in_data, ES_GSTART, isGrad));
	ERR(ES_SetParamVisible(in_data, ES_GEND,   isGrad));

	return err;
}

static PF_Err
UserChangedParam (
	PF_InData						*in_data,
	PF_OutData						*out_data,
	PF_ParamDef						*params[],
	const PF_UserChangedParamExtra	*which )
{
	PF_Err	err = PF_Err_NONE;
	// Only the two supervised popups need UI work. Their VALUES already drive
	// PreRender, so the frame re-renders on its own - this is UI-only.
	if (which->param_index == ES_TYPE || which->param_index == ES_FILL)
		ERR(ES_SyncUI(in_data, params));
	return err;
}

/* =========================================================================
   SmartFX phase 1: PRE-RENDER (region negotiation only).
   ========================================================================= */

static PF_Err
PreRender (
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_PreRenderExtra	*extra )
{
	PF_Err				err = PF_Err_NONE, err2 = PF_Err_NONE;
	PF_RenderRequest	req = extra->input->output_request;
	PF_CheckoutResult	in_result;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	PF_Handle infoH = suites.HandleSuite1()->host_new_handle(sizeof(ESInfo));
	if (!infoH) return PF_Err_OUT_OF_MEMORY;

	ESInfo *info = reinterpret_cast<ESInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!info) { suites.HandleSuite1()->host_dispose_handle(infoH); return PF_Err_OUT_OF_MEMORY; }

	extra->output->pre_render_data = infoH;
	AEFX_CLR_STRUCT(*info);

	// Distances are authored at full res; scale them to the current resolution.
	PF_FpLong dsX = (PF_FpLong)in_data->downsample_x.num / in_data->downsample_x.den;
	PF_FpLong dsY = (PF_FpLong)in_data->downsample_y.num / in_data->downsample_y.den;
	PF_FpLong ds  = (dsX + dsY) * 0.5;

	#define ES_CHECKOUT(idx, var)											\
		PF_ParamDef var;													\
		AEFX_CLR_STRUCT(var);												\
		ERR(PF_CHECKOUT_PARAM(in_data, (idx), in_data->current_time,		\
							  in_data->time_step, in_data->time_scale, &var));

	{
		ES_CHECKOUT(ES_TYPE,		type_p);
		ES_CHECKOUT(ES_DIRECTION,	dir_p);
		ES_CHECKOUT(ES_LENGTH,		len_p);
		ES_CHECKOUT(ES_SOURCE,		src_p);
		ES_CHECKOUT(ES_LENGTH_PCT,	pct_p);
		ES_CHECKOUT(ES_FADE_IN,		fin_p);
		ES_CHECKOUT(ES_FADE_OUT,	fout_p);
		ES_CHECKOUT(ES_COLOR,		col_p);
		ES_CHECKOUT(ES_FILL,		fill_p);
		ES_CHECKOUT(ES_COLOR2,		col2_p);
		ES_CHECKOUT(ES_GSTART,		gs_p);
		ES_CHECKOUT(ES_GEND,		ge_p);
		ES_CHECKOUT(ES_TINT,		tint_p);
		ES_CHECKOUT(ES_OPACITY,		op_p);
		ES_CHECKOUT(ES_THRESHOLD,	thr_p);

		if (!err) {
			info->type		= type_p.u.pd.value;
			info->angle		= dir_p.u.ad.value / 65536.0;
			info->length	= len_p.u.fs_d.value * ds;
			info->srcX		= src_p.u.td.x_value / 65536.0;
			info->srcY		= src_p.u.td.y_value / 65536.0;
			info->lengthPct	= pct_p.u.fs_d.value / 100.0;
			info->fadeIn	= fin_p.u.fs_d.value / 100.0;
			info->fadeOut	= fout_p.u.fs_d.value / 100.0;
			info->color[0]	= col_p.u.cd.value.red   / 255.0;
			info->color[1]	= col_p.u.cd.value.green / 255.0;
			info->color[2]	= col_p.u.cd.value.blue  / 255.0;
			info->fill		= fill_p.u.pd.value;
			info->color2[0]	= col2_p.u.cd.value.red   / 255.0;
			info->color2[1]	= col2_p.u.cd.value.green / 255.0;
			info->color2[2]	= col2_p.u.cd.value.blue  / 255.0;
			info->gsx		= gs_p.u.td.x_value / 65536.0;
			info->gsy		= gs_p.u.td.y_value / 65536.0;
			info->gex		= ge_p.u.td.x_value / 65536.0;
			info->gey		= ge_p.u.td.y_value / 65536.0;
			info->tint		= tint_p.u.fs_d.value / 100.0;
			info->opacity	= op_p.u.fs_d.value / 100.0;
			info->threshold	= thr_p.u.fs_d.value / 100.0;
		}

		ERR2(PF_CHECKIN_PARAM(in_data, &type_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &dir_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &len_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &src_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &pct_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &fin_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &fout_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &col_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &fill_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &col2_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &gs_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &ge_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &tint_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &op_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &thr_p));
	}
	#undef ES_CHECKOUT

	// How far the shadow can reach, in render px. Length bounds BOTH types:
	// directional runs exactly Length, and radial's per-pixel length (pct% of the
	// distance to the source) is CAPPED at Length in SmartRender. So Length alone
	// is the exact reach - no separate "grow bounds" control is needed (an earlier
	// one only ever over-grew the buffer, since the shadow can never exceed L).
	PF_FpLong reach = info->length;

	const PF_LRect reqRect = extra->input->output_request.rect;

	A_long grow = (A_long)ceil(reach + 1.0);
	if (grow < 0) grow = 0;

	req.rect.left	-= grow;
	req.rect.top	-= grow;
	req.rect.right	+= grow;
	req.rect.bottom	+= grow;
	req.preserve_rgb_of_zero_alpha = TRUE;
	req.field = PF_Field_FRAME;

	ERR(extra->cb->checkout_layer(	in_data->effect_ref,
									ES_INPUT, ES_INPUT, &req,
									in_data->current_time, in_data->time_step,
									in_data->time_scale, &in_result));

	if (!err) {
		// Empty input (a transparent frame, a text layer with no visible glyphs)
		// has nothing to shadow. Growing an empty rect would invent a box the
		// size of the growth and declare we cover it - the phantom-rectangle bug
		// from Buildable Stroke. Stay empty instead.
		const bool inputEmpty =
			(in_result.result_rect.right  <= in_result.result_rect.left) ||
			(in_result.result_rect.bottom <= in_result.result_rect.top);

		if (inputEmpty) {
			extra->output->result_rect     = in_result.result_rect;
			extra->output->max_result_rect = in_result.max_result_rect;
			info->inRect  = in_result.result_rect;
			info->outRect = in_result.result_rect;
			info->pad     = 0;
			extra->output->solid = FALSE;
			suites.HandleSuite1()->host_unlock_handle(infoH);
			return err;
		}

		// max_result_rect MAY exceed the request (it grows the layer bounds so
		// the shadow isn't clipped). result_rect MUST stay within the request
		// (per-tile promise). See the Buildable Stroke notes.
		PF_LRect m = in_result.max_result_rect;
		m.left -= grow;  m.top -= grow;  m.right += grow;  m.bottom += grow;
		extra->output->max_result_rect = m;

		PF_LRect r = in_result.result_rect;
		r.left -= grow;  r.top -= grow;  r.right += grow;  r.bottom += grow;
		if (r.left   < reqRect.left)	r.left   = reqRect.left;
		if (r.top    < reqRect.top)		r.top    = reqRect.top;
		if (r.right  > reqRect.right)	r.right  = reqRect.right;
		if (r.bottom > reqRect.bottom)	r.bottom = reqRect.bottom;
		if (r.right  < r.left)			r.right  = r.left;
		if (r.bottom < r.top)			r.bottom = r.top;
		extra->output->result_rect = r;

		// Stash the DECLARED result_rect (not the request) so SmartRender maps
		// output pixels to layer/input coords correctly - the output world
		// covers exactly what we promised here.
		info->inRect	= in_result.result_rect;
		info->outRect	= r;
		info->pad		= grow;

		extra->output->solid = FALSE;

#if ES_GPU_RENDER
		// Opt THIS frame into GPU rendering. Only takes effect when the global
		// SUPPORTS_GPU_RENDER_F32 flag is also set and AE has a supported (CUDA)
		// device; otherwise AE falls back to PF_Cmd_SMART_RENDER (CPU). While
		// ES_GPU_RENDER is 0 we never set this, so the GPU path stays dormant.
		extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
#endif
	}

	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}

/* =========================================================================
   Pixel I/O (any bit depth), and bilinear sampling of the cropped buffers.
   ========================================================================= */

static float
ES_ReadAlpha (const PF_EffectWorld *w, A_long bd, A_long u, A_long v)
{
	if (u < 0 || v < 0 || u >= w->width || v >= w->height) return 0.0f;
	const char *row = (const char*)w->data + (size_t)v * w->rowbytes;
	switch (bd) {
		case 32: return ((const PF_PixelFloat*)row)[u].alpha;
		case 16: return ((const PF_Pixel16*)row)[u].alpha / (float)PF_MAX_CHAN16;
		default: return ((const PF_Pixel8*)row)[u].alpha  / 255.0f;
	}
}

static void
ES_ReadRGBA (const PF_EffectWorld *w, A_long bd, A_long u, A_long v, float *rgba)
{
	rgba[0] = rgba[1] = rgba[2] = rgba[3] = 0.0f;
	if (u < 0 || v < 0 || u >= w->width || v >= w->height) return;
	const char *row = (const char*)w->data + (size_t)v * w->rowbytes;
	switch (bd) {
		case 32: {
			const PF_PixelFloat *p = &((const PF_PixelFloat*)row)[u];
			rgba[0]=p->red; rgba[1]=p->green; rgba[2]=p->blue; rgba[3]=p->alpha;
		} break;
		case 16: {
			const PF_Pixel16 *p = &((const PF_Pixel16*)row)[u];
			rgba[0]=p->red/(float)PF_MAX_CHAN16; rgba[1]=p->green/(float)PF_MAX_CHAN16;
			rgba[2]=p->blue/(float)PF_MAX_CHAN16; rgba[3]=p->alpha/(float)PF_MAX_CHAN16;
		} break;
		default: {
			const PF_Pixel8 *p = &((const PF_Pixel8*)row)[u];
			rgba[0]=p->red/255.0f; rgba[1]=p->green/255.0f;
			rgba[2]=p->blue/255.0f; rgba[3]=p->alpha/255.0f;
		} break;
	}
}

static void
ES_WriteRGBA (PF_EffectWorld *w, A_long bd, A_long x, A_long y, const float *rgbaIn)
{
	char *row = (char*)w->data + (size_t)y * w->rowbytes;

	// A NaN slips through a naive clamp and an infinity clamps to FULL - either
	// turns a stray value into a solid opaque block. Sanitize alpha to [0,1].
	float rgba[4];
	for (A_long c = 0; c < 4; c++) { float v = rgbaIn[c]; if (!(v == v)) v = 0.0f; rgba[c] = v; }
	if (!(rgba[3] >= 0.0f))	rgba[3] = 0.0f;
	if (rgba[3] > 1.0f)		rgba[3] = 1.0f;

	#define ES_CL(v, mx) ((v) < 0 ? 0 : ((v) > (mx) ? (mx) : (v)))
	switch (bd) {
		case 32: {
			PF_PixelFloat *p = &((PF_PixelFloat*)row)[x];
			p->red=rgba[0]; p->green=rgba[1]; p->blue=rgba[2]; p->alpha=rgba[3];
		} break;
		case 16: {
			PF_Pixel16 *p = &((PF_Pixel16*)row)[x];
			p->red  =(A_u_short)ES_CL(rgba[0]*PF_MAX_CHAN16+0.5f,(float)PF_MAX_CHAN16);
			p->green=(A_u_short)ES_CL(rgba[1]*PF_MAX_CHAN16+0.5f,(float)PF_MAX_CHAN16);
			p->blue =(A_u_short)ES_CL(rgba[2]*PF_MAX_CHAN16+0.5f,(float)PF_MAX_CHAN16);
			p->alpha=(A_u_short)ES_CL(rgba[3]*PF_MAX_CHAN16+0.5f,(float)PF_MAX_CHAN16);
		} break;
		default: {
			PF_Pixel8 *p = &((PF_Pixel8*)row)[x];
			p->red  =(A_u_char)ES_CL(rgba[0]*255.0f+0.5f,255.0f);
			p->green=(A_u_char)ES_CL(rgba[1]*255.0f+0.5f,255.0f);
			p->blue =(A_u_char)ES_CL(rgba[2]*255.0f+0.5f,255.0f);
			p->alpha=(A_u_char)ES_CL(rgba[3]*255.0f+0.5f,255.0f);
		} break;
	}
	#undef ES_CL
}

/* Bilinear sample of a cropped scalar buffer at fractional cropped coords.
   Clamped to the buffer edge - safe because the work rect is (shape bbox + pad),
   so its border is >= pad from any shape pixel and therefore already ~0. */
static float
ES_SampleF (const float *buf, A_long CW, A_long CH, float fx, float fy)
{
	if (fx < 0.0f) fx = 0.0f; else if (fx > CW - 1) fx = (float)(CW - 1);
	if (fy < 0.0f) fy = 0.0f; else if (fy > CH - 1) fy = (float)(CH - 1);
	A_long x0 = (A_long)fx, y0 = (A_long)fy;
	A_long x1 = (x0 + 1 < CW) ? x0 + 1 : x0;
	A_long y1 = (y0 + 1 < CH) ? y0 + 1 : y0;
	float tx = fx - x0, ty = fy - y0;
	float a0 = buf[y0*CW+x0] + (buf[y0*CW+x1] - buf[y0*CW+x0]) * tx;
	float a1 = buf[y1*CW+x0] + (buf[y1*CW+x1] - buf[y1*CW+x0]) * tx;
	return a0 + (a1 - a0) * ty;
}

static void
ES_SampleRGB (const float *buf, A_long CW, A_long CH, float fx, float fy, float *out)
{
	if (fx < 0.0f) fx = 0.0f; else if (fx > CW - 1) fx = (float)(CW - 1);
	if (fy < 0.0f) fy = 0.0f; else if (fy > CH - 1) fy = (float)(CH - 1);
	A_long x0 = (A_long)fx, y0 = (A_long)fy;
	A_long x1 = (x0 + 1 < CW) ? x0 + 1 : x0;
	A_long y1 = (y0 + 1 < CH) ? y0 + 1 : y0;
	float tx = fx - x0, ty = fy - y0;
	for (A_long c = 0; c < 3; c++) {
		float a0 = buf[(y0*CW+x0)*3+c] + (buf[(y0*CW+x1)*3+c] - buf[(y0*CW+x0)*3+c]) * tx;
		float a1 = buf[(y1*CW+x0)*3+c] + (buf[(y1*CW+x1)*3+c] - buf[(y1*CW+x0)*3+c]) * tx;
		out[c] = a0 + (a1 - a0) * ty;
	}
}

/* Split [0,n) into contiguous ranges, one per thread, and run fn(lo,hi) on each.
   fn must touch only the rows its [lo,hi) owns. The march is embarrassingly
   parallel over OUTPUT ROWS: each row writes its own disjoint output pixels and
   only READS the shared input / cropped buffers, so no locks are needed.
   Small heights, or a single CPU, run inline. Capped at 8 threads so this stays
   sane alongside AE's multi-frame rendering (which already uses cores). */
template <class F>
static void
ES_ParallelRanges (A_long n, F fn)
{
	if (n <= 0) return;
	A_long hw = (A_long)std::thread::hardware_concurrency();
	if (hw < 1) hw = 1;
	A_long T = (hw < 8) ? hw : 8;
	if (n < 8 || T <= 1) { fn(0, n); return; }
	if (T > n) T = n;

	const A_long chunk = (n + T - 1) / T;
	std::vector<std::thread> pool;
	pool.reserve((size_t)(T - 1));
	for (A_long t = 1; t < T; t++) {
		A_long lo = t * chunk, hi = lo + chunk;
		if (hi > n) hi = n;
		if (lo >= hi) break;
		pool.emplace_back([=]{ fn(lo, hi); });
	}
	fn(0, (chunk < n) ? chunk : n);		// main thread takes the first block
	for (auto &th : pool) th.join();
}

/* =========================================================================
   SmartFX phase 2: SMART RENDER (the march).
   ========================================================================= */

static PF_Err
SmartRender (
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_SmartRenderExtra	*extra )
{
	PF_Err				err = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);
	PF_EffectWorld		*inputP = NULL, *outputP = NULL;

	PF_Handle infoH = reinterpret_cast<PF_Handle>(extra->input->pre_render_data);
	ESInfo *info = reinterpret_cast<ESInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!info) return PF_Err_BAD_CALLBACK_PARAM;

	ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, ES_INPUT, &inputP));
	ERR(extra->cb->checkout_output(in_data->effect_ref, &outputP));

	if (!err && inputP && outputP) {
		const A_long	bd	= extra->input->bitdepth;
		const A_long	W	= outputP->width;
		const A_long	H	= outputP->height;
		const A_long	pad	= info->pad;
		const A_long	PW	= W + 2 * pad;
		const A_long	PH	= H + 2 * pad;

		const A_long	offX = info->inRect.left - info->outRect.left;
		const A_long	offY = info->inRect.top  - info->outRect.top;
		const A_long	oox  = info->outRect.left;		// output (x,y) -> layer (x+oox, y+ooy)
		const A_long	ooy  = info->outRect.top;

		#define ES_IN_X(px)	((px) - pad - offX)
		#define ES_IN_Y(py)	((py) - pad - offY)

		const float thr = (float)info->threshold;

		try {
			// --- pass 1: find the shape bbox in padded space -----------------
			A_long minX = PW, minY = PH, maxX = -1, maxY = -1;
			for (A_long py = 0; py < PH; py++)
				for (A_long px = 0; px < PW; px++)
					if (ES_ReadAlpha(inputP, bd, ES_IN_X(px), ES_IN_Y(py)) >= thr) {
						if (px < minX) minX = px;
						if (px > maxX) maxX = px;
						if (py < minY) minY = py;
						if (py > maxY) maxY = py;
					}

			// Nothing above threshold -> no shadow anywhere; copy input through.
			if (maxX < 0) {
				for (A_long y = 0; y < H; y++)
					for (A_long x = 0; x < W; x++) {
						float src[4];
						ES_ReadRGBA(inputP, bd, x - offX, y - offY, src);
						ES_WriteRGBA(outputP, bd, x, y, src);
					}
				suites.HandleSuite1()->host_unlock_handle(infoH);
				return err;
			}

			// --- crop every buffer to (bbox + pad) ---------------------------
			const A_long wx0 = MAX(0, minX - pad);
			const A_long wy0 = MAX(0, minY - pad);
			const A_long wx1 = MIN(PW - 1, maxX + pad);
			const A_long wy1 = MIN(PH - 1, maxY + pad);
			const A_long CW  = wx1 - wx0 + 1;
			const A_long CH  = wy1 - wy0 + 1;
			const A_long N   = CW * CH;

			// output (x,y) -> cropped coords (cx,cy)
			#define ES_CX(x)	((x) + pad - wx0)
			#define ES_CY(y)	((y) + pad - wy0)

			const bool needCol = (info->tint < 1.0);	// object colour, for tint

			std::vector<float> alphaBuf(N);
			std::vector<float> colBuf;
			if (needCol) colBuf.assign((size_t)N * 3, 0.0f);

			for (A_long cy = 0; cy < CH; cy++)
				for (A_long cx = 0; cx < CW; cx++) {
					A_long iu = ES_IN_X(cx + wx0), iv = ES_IN_Y(cy + wy0);
					alphaBuf[cy*CW+cx] = ES_ReadAlpha(inputP, bd, iu, iv);
					if (needCol) {
						float rgba[4];
						ES_ReadRGBA(inputP, bd, iu, iv, rgba);
						colBuf[(cy*CW+cx)*3+0] = rgba[0];
						colBuf[(cy*CW+cx)*3+1] = rgba[1];
						colBuf[(cy*CW+cx)*3+2] = rgba[2];
					}
				}

			// Precompute constants shared by every pixel.
			const int   type   = (int)info->type;
			const float rad    = (float)(info->angle * ES_PI / 180.0);
			const float ddx    = sinf(rad);			// directional projection dir
			const float ddy    = -cosf(rad);
			const float dirL   = (float)info->length;
			const float pct    = (float)info->lengthPct;
			const float srcX   = (float)info->srcX;
			const float srcY   = (float)info->srcY;
			const float fadeIn = (float)info->fadeIn;
			const float fadeOut= (float)info->fadeOut;
			const float tint   = (float)info->tint;
			const float op     = (float)info->opacity;
			const bool  isGrad   = (info->fill == ES_FILL_LINEAR ||
									info->fill == ES_FILL_RADIAL);
			const bool  isRadial = (info->fill == ES_FILL_RADIAL);
			const float gdx    = (float)(info->gex - info->gsx);
			const float gdy    = (float)(info->gey - info->gsy);
			const float gden   = gdx*gdx + gdy*gdy;
			const float ginv   = (gden > 1e-6f) ? 1.0f/gden : 0.0f;
			// Radial: 1/radius = 1/sqrt(|end-start|^2). Zero radius -> t clamps
			// to 1 everywhere but the exact center (fully col2).
			const float rinv   = (gden > 1e-6f) ? 1.0f/sqrtf(gden) : 0.0f;
			const float col0   = (float)info->color[0];
			const float col1   = (float)info->color[1];
			const float col2   = (float)info->color[2];
			const float colE0  = (float)info->color2[0];
			const float colE1  = (float)info->color2[1];
			const float colE2  = (float)info->color2[2];
			const float STEP   = 0.5f;					// march sample spacing, px

			// --- march every output pixel (threaded over rows) ---------------
			auto renderRows = [&](A_long y0, A_long y1) {
			for (A_long y = y0; y < y1; y++) {
				const A_long py = y + pad;
				const bool rowIn = (py >= wy0 && py <= wy1);
				for (A_long x = 0; x < W; x++) {
					float src[4];
					ES_ReadRGBA(inputP, bd, x - offX, y - offY, src);

					const A_long px = x + pad;
					if (!rowIn || px < wx0 || px > wx1) {
						ES_WriteRGBA(outputP, bd, x, y, src);	// beyond any reach
						continue;
					}

					const float cx = (float)ES_CX(x);
					const float cy = (float)ES_CY(y);
					const float Lx = (float)(x + oox);
					const float Ly = (float)(y + ooy);

					// direction + length for this pixel
					float dx, dy, L;
					if (type == ES_TYPE_DIR) {
						dx = ddx; dy = ddy; L = dirL;
					} else {
						float vx = Lx - srcX, vy = Ly - srcY;
						float dist = sqrtf(vx*vx + vy*vy);
						if (dist < 1e-3f) { dx = dy = 0.0f; L = 0.0f; }
						else {
							float inv = 1.0f / dist;
							if (type == ES_TYPE_INVRADIAL) { dx = -vx*inv; dy = -vy*inv; }
							else                           { dx =  vx*inv; dy =  vy*inv; }
							L = pct * dist;			// % of distance to source ...
							if (L > dirL) L = dirL;	// ... capped by Length (max reach)
						}
					}

					// Clip the march length to where the ray leaves the cropped
					// buffer - beyond that there is no shape to sample.
					float tMax = L;
					if (dx >  1e-6f) { float tb = cx / dx;                  if (tb < tMax) tMax = tb; }
					else if (dx < -1e-6f) { float tb = (cx-(CW-1)) / dx;    if (tb < tMax) tMax = tb; }
					if (dy >  1e-6f) { float tb = cy / dy;                  if (tb < tMax) tMax = tb; }
					else if (dy < -1e-6f) { float tb = (cy-(CH-1)) / dy;    if (tb < tMax) tMax = tb; }
					if (tMax < 0.0f) tMax = 0.0f;

					// t = 0 term
					float cov = ES_SampleF(&alphaBuf[0], CW, CH, cx, cy);
					float fadeDist = (cov >= thr) ? 0.0f : ES_INF;
					float hitX = cx, hitY = cy;

					for (float t = STEP; t <= tMax + 1e-4f; t += STEP) {
						float sx = cx - dx * t, sy = cy - dy * t;
						float a = ES_SampleF(&alphaBuf[0], CW, CH, sx, sy);
						if (a > cov) cov = a;
						if (fadeDist == ES_INF && a >= thr) { fadeDist = t; hitX = sx; hitY = sy; }
					}

					if (cov <= 0.0f) { ES_WriteRGBA(outputP, bd, x, y, src); continue; }
					if (fadeDist == ES_INF) fadeDist = (L > 0.0f) ? L : 0.0f;

					// fade ramp along the length
					float s = (L > 1e-6f) ? (fadeDist / L) : 0.0f;
					if (s < 0.0f) s = 0.0f; else if (s > 1.0f) s = 1.0f;
					float ramp = fadeIn + (fadeOut - fadeIn) * s;

					// fill colour: solid, or lerp along the gradient axis
					float fr = col0, fg = col1, fb = col2;
					if (isGrad) {
						const float rx = Lx - (float)info->gsx;
						const float ry = Ly - (float)info->gsy;
						float tg = isRadial
								 ? sqrtf(rx*rx + ry*ry) * rinv
								 : (rx * gdx + ry * gdy) * ginv;
						if (tg < 0.0f) tg = 0.0f; else if (tg > 1.0f) tg = 1.0f;
						fr = col0 + (colE0 - col0) * tg;
						fg = col1 + (colE1 - col1) * tg;
						fb = col2 + (colE2 - col2) * tg;
					}

					// tint toward the shadow hue (tint=1 flat colour, tint=0 keeps
					// the object's own colour sampled at the ray's hit point)
					float shR = fr, shG = fg, shB = fb;
					if (needCol) {
						float oc[3];
						ES_SampleRGB(&colBuf[0], CW, CH, hitX, hitY, oc);
						shR = oc[0]*(1.0f-tint) + fr*tint;
						shG = oc[1]*(1.0f-tint) + fg*tint;
						shB = oc[2]*(1.0f-tint) + fb*tint;
					}

					float shA = cov * ramp * op;
					if (shA < 0.0f) shA = 0.0f; else if (shA > 1.0f) shA = 1.0f;

					// composite: source art OVER its shadow (straight alpha)
					float sa = src[3];
					float outA = sa + shA * (1.0f - sa);
					float out[4];
					if (outA > 1e-6f) {
						float inv = 1.0f / outA;
						out[0] = (src[0]*sa + shR*shA*(1.0f-sa)) * inv;
						out[1] = (src[1]*sa + shG*shA*(1.0f-sa)) * inv;
						out[2] = (src[2]*sa + shB*shA*(1.0f-sa)) * inv;
					} else {
						out[0] = out[1] = out[2] = 0.0f;
					}
					out[3] = outA;
					ES_WriteRGBA(outputP, bd, x, y, out);
				}
			}
			};	// end renderRows lambda
			ES_ParallelRanges(H, renderRows);

			#undef ES_CX
			#undef ES_CY
		}
		catch (std::bad_alloc &) {
			err = PF_Err_OUT_OF_MEMORY;
		}
		#undef ES_IN_X
		#undef ES_IN_Y
	}

	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}

/* =========================================================================
   GPU path (CUDA only) - SCAFFOLD stage.

   Mirrors the SDK's SDK_Invert_ProcAmp split:
     GPU_DEVICE_SETUP   - per CUDA device AE exposes. The kernel is statically
                          linked, so we just confirm we support this device.
     GPU_DEVICE_SETDOWN - matching teardown (nothing to free for CUDA).
     SMART_RENDER_GPU   - per frame: get device pointers for input/output worlds
                          and launch. Reached only when PreRender set
                          GPU_RENDER_POSSIBLE (gated by ES_GPU_RENDER).
   PreRender is shared with the CPU path, so info->{inRect,outRect,pad,...} are
   already populated.
   ========================================================================= */

static PF_Err
GPUDeviceSetup (
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_GPUDeviceSetupExtra	*extra )
{
	// CUDA (Windows): kernel is statically linked, nothing to build here - just
	// claim F32 support. Any framework we don't handle is left on the CPU path.
	if (extra->input->what_gpu == PF_GPU_Framework_CUDA) {
		out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#if HAS_METAL
	// Metal (macOS): compile the MSL library and build the pipeline states once
	// per device, stash them in gpu_data. On a compile error, surface Metal's
	// own message in AE's error dialog so it can be read directly.
	else if (extra->input->what_gpu == PF_GPU_Framework_METAL) {
		PF_GPUDeviceSuite1 *gpu = NULL;
		if (in_data->pica_basicP->AcquireSuite(kPFGPUDeviceSuite,
				kPFGPUDeviceSuiteVersion1, (const void**)&gpu) || !gpu)
			return PF_Err_BAD_CALLBACK_PARAM;

		PF_GPUDeviceInfo info;
		AEFX_CLR_STRUCT(info);
		PF_Err e = gpu->GetDeviceInfo(in_data->effect_ref,
					extra->input->device_index, &info);
		in_data->pica_basicP->ReleaseSuite(kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
		if (e) return e;

		void *metalData = NULL;
		char  errBuf[512] = {0};
		if (!ES_MetalCompile(info.devicePV, &metalData, errBuf, sizeof(errBuf))) {
			PF_STRCPY(out_data->return_msg, "ExtendedShadow Metal build failed: ");
			strncat(out_data->return_msg, errBuf,
					sizeof(out_data->return_msg) - strlen(out_data->return_msg) - 1);
			out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}
		extra->output->gpu_data = metalData;			// opaque; freed in Setdown
		out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
	return PF_Err_NONE;
}

static PF_Err
GPUDeviceSetdown (
	PF_InData					*in_data,
	PF_OutData					*out_data,
	PF_GPUDeviceSetdownExtra	*extra )
{
#if HAS_METAL
	// Release the Metal pipeline states built in GPUDeviceSetup.
	if (extra->input->what_gpu == PF_GPU_Framework_METAL && extra->input->gpu_data) {
		ES_MetalDispose(const_cast<void*>(extra->input->gpu_data));
	}
#endif
	// CUDA kernel is statically linked; nothing device-specific to release.
	return PF_Err_NONE;
}

#if HAS_CUDA || HAS_METAL
static PF_Err
SmartRenderGPU (
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_SmartRenderExtra	*extra )
{
	PF_Err				err		= PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);
	PF_EffectWorld		*inputP = NULL, *outputP = NULL;

	// pre_render_data is the PF_Handle from PreRender - it must be LOCKED to get
	// the ESInfo pointer, exactly as the CPU SmartRender does (treating the handle
	// as the pointer gives garbage rects and the whole layer disappears).
	PF_Handle infoH = reinterpret_cast<PF_Handle>(extra->input->pre_render_data);
	ESInfo *info = reinterpret_cast<ESInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!info) return PF_Err_BAD_CALLBACK_PARAM;

	PF_GPUDeviceSuite1 *gpu = NULL;
	err = in_data->pica_basicP->AcquireSuite(
			kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, (const void**)&gpu);
	if (err || !gpu) {
		suites.HandleSuite1()->host_unlock_handle(infoH);
		return err ? err : PF_Err_BAD_CALLBACK_PARAM;
	}

	ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, ES_INPUT, &inputP));
	ERR(extra->cb->checkout_output(in_data->effect_ref, &outputP));

	if (!err && inputP && outputP) {
		void *src_mem = NULL, *dst_mem = NULL;
		ERR(gpu->GetGPUWorldData(in_data->effect_ref, inputP,  &src_mem));
		ERR(gpu->GetGPUWorldData(in_data->effect_ref, outputP, &dst_mem));

		if (!err) {
			const int bytesPerPixel = 16;			// GPU_BGRA128 = 4 x float32

			ESGpuParams p;
			memset(&p, 0, sizeof(p));
			p.type      = (int)info->type;
			float radK  = (float)(info->angle * ES_PI / 180.0);
			p.ddx       = sinf(radK);
			p.ddy       = -cosf(radK);
			p.length    = (float)info->length;
			p.srcX      = (float)info->srcX;
			p.srcY      = (float)info->srcY;
			p.lengthPct = (float)info->lengthPct;
			p.fadeIn    = (float)info->fadeIn;
			p.fadeOut   = (float)info->fadeOut;
			p.tint      = (float)info->tint;
			p.opacity   = (float)info->opacity;
			p.threshold = (float)info->threshold;
			p.fill      = (int)info->fill;
			p.col[0]    = (float)info->color[0];
			p.col[1]    = (float)info->color[1];
			p.col[2]    = (float)info->color[2];
			p.col2[0]   = (float)info->color2[0];
			p.col2[1]   = (float)info->color2[1];
			p.col2[2]   = (float)info->color2[2];
			p.gsx       = (float)info->gsx;
			p.gsy       = (float)info->gsy;
			p.gex       = (float)info->gex;
			p.gey       = (float)info->gey;
			p.offX      = info->inRect.left - info->outRect.left;
			p.offY      = info->inRect.top  - info->outRect.top;
			p.oox       = info->outRect.left;
			p.ooy       = info->outRect.top;
			p.inW       = inputP->width;
			p.inH       = inputP->height;
			p.W         = outputP->width;
			p.H         = outputP->height;
			p.srcPitch  = inputP->rowbytes  / bytesPerPixel;
			p.dstPitch  = outputP->rowbytes / bytesPerPixel;
			p.step      = 0.5f;
			p.pad       = (int)info->pad;
			// wx0..wy1 are filled by the launcher after the on-device bbox pass.

#if HAS_CUDA
			if (extra->input->what_gpu == PF_GPU_Framework_CUDA) {
				ES_March_CUDA((const float*)src_mem, (float*)dst_mem, p);
				if (cudaPeekAtLastError() != cudaSuccess)
					err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}
#endif
#if HAS_METAL
			if (extra->input->what_gpu == PF_GPU_Framework_METAL) {
				// Metal needs the device + command queue (CUDA used the default
				// context). gpu_data holds the pipeline states built in setup.
				PF_GPUDeviceInfo devInfo;
				AEFX_CLR_STRUCT(devInfo);
				ERR(gpu->GetDeviceInfo(in_data->effect_ref,
						extra->input->device_index, &devInfo));
				if (!err) {
					if (!ES_March_Metal(devInfo.devicePV, devInfo.command_queuePV,
							const_cast<void*>(extra->input->gpu_data),
							src_mem, dst_mem, p))
						err = PF_Err_INTERNAL_STRUCT_DAMAGED;
				}
			}
#endif
		}
	}

	in_data->pica_basicP->ReleaseSuite(kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}
#endif // HAS_CUDA || HAS_METAL

/* =========================================================================
   Classic render path (Premiere / legacy): pass through. AE uses SmartFX.
   ========================================================================= */

static PF_Err
Render (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err err = PF_Err_NONE;
	ERR(PF_COPY(&params[ES_INPUT]->u.ld, output, NULL, NULL));
	return err;
}

/* =========================================================================
   Registration + entry point.
   ========================================================================= */

extern "C" DllExport
PF_Err PluginDataEntryFunction2(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB2 inPluginDataCallBackPtr,
	SPBasicSuite* inSPBasicSuitePtr,
	const char* inHostName,
	const char* inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	result = PF_REGISTER_EFFECT_EXT2(
		inPtr,
		inPluginDataCallBackPtr,
		"Extended Shadow",			// Name
		"aldai ExtendedShadow",		// Match Name
		"Learning",					// Category
		AE_RESERVED_INFO,			// Reserved Info
		"EffectMain",				// Entry point
		"https://www.adobe.com");	// support URL

	return result;
}

PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err err = PF_Err_NONE;
	try {
		switch (cmd) {
			case PF_Cmd_ABOUT:
				err = About(in_data, out_data, params, output);
				break;
			case PF_Cmd_GLOBAL_SETUP:
				err = GlobalSetup(in_data, out_data, params, output);
				break;
			case PF_Cmd_PARAMS_SETUP:
				err = ParamsSetup(in_data, out_data, params, output);
				break;
			case PF_Cmd_USER_CHANGED_PARAM:		// Type / Fill popup changed
				err = UserChangedParam(in_data, out_data, params,
						reinterpret_cast<const PF_UserChangedParamExtra*>(extra));
				break;
			case PF_Cmd_UPDATE_PARAMS_UI:		// sync visibility on load
				err = ES_SyncUI(in_data, params);
				break;
			case PF_Cmd_RENDER:
				err = Render(in_data, out_data, params, output);
				break;
			case PF_Cmd_SMART_PRE_RENDER:
				err = PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra));
				break;
			case PF_Cmd_SMART_RENDER:
				err = SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra));
				break;
			case PF_Cmd_GPU_DEVICE_SETUP:
				err = GPUDeviceSetup(in_data, out_data,
						reinterpret_cast<PF_GPUDeviceSetupExtra*>(extra));
				break;
			case PF_Cmd_GPU_DEVICE_SETDOWN:
				err = GPUDeviceSetdown(in_data, out_data,
						reinterpret_cast<PF_GPUDeviceSetdownExtra*>(extra));
				break;
#if HAS_CUDA || HAS_METAL
			case PF_Cmd_SMART_RENDER_GPU:
				err = SmartRenderGPU(in_data, out_data,
						reinterpret_cast<PF_SmartRenderExtra*>(extra));
				break;
#endif
		}
	}
	catch(PF_Err &thrown_err){
		err = thrown_err;
	}
	return err;
}
