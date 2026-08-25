/*
	ExtendedShadow.h

	Fourth learning plugin, ported from python-proto/long_shadow/ (ls_step1..3).
	A "long shadow": the layer's alpha silhouette projected along a direction
	(or radially from a point) and filled with a colour / gradient, fading along
	its length. Named "Extended Shadow" to avoid confusion with Creative Dojo's
	Long Shadow.

	Like Buildable Stroke this is NOT a point operation:
	  - output is LARGER than input   -> SmartFX buffer expansion is mandatory
	  - it re-reads the source over a whole region per output pixel (the march)

	THE CORE IDEA (see the Python prototype for the derivation):
	  Most long-shadow plugins project a BINARY silhouette (is there an opaque
	  pixel within L?) and then bolt on Supersampling + Post-Smooth to hide the
	  stair-steps. We instead read the source's SOFT alpha:
	      cov(p) = max over t in [0, L] of  alpha_src(p - t * d)    (bilinear)
	  Because alpha is already anti-aliased, the shadow is smooth by construction
	  - no supersampling. Same "re-read the existing soft edge" principle as
	  Chromatic Aberration and Buildable Stroke.

	SHADOW TYPES:
	  Directional      - d is constant = (sin a, -cos a) from the angle dial.
	  Radial           - d(p) = normalize(p - source); Length is a % of the
	                     pixel's distance to the source.
	  Inverse Radial   - d(p) = normalize(source - p) (casts toward the point).

	FADE + COLOUR (ls_step3):
	  The march also records fadeDist = the distance at which each ray first hits
	  the object, and the source colour there. Then:
	      s    = clamp(fadeDist / L)
	      ramp = lerp(fadeIn, fadeOut, s)          (fade along the length)
	      fill = solid colour, or a linear gradient between two colours
	      rgb  = lerp(objColour, fill, tint)       (tint toward the shadow hue)
	      a    = cov * ramp * opacity
	  Finally the source art composites OVER its shadow.

	STAGE 1 (this file): CPU only, single shadow, the march for every type. The
	width-independent log-doubling fast path (directional) and the CUDA port are
	later stages, exactly as Buildable Stroke was staged.

	NOT PLANNED (unless requested): Shadow Texture, Object Color (recolouring the
	source art itself). Soft Shadow (progressive blur) is a possible v2.
*/

#pragma once

#ifndef EXTENDEDSHADOW_H
#define EXTENDEDSHADOW_H

typedef unsigned char		u_char;
typedef unsigned short		u_short;
typedef unsigned short		u_int16;
typedef unsigned long		u_long;
typedef short int			int16;
#define PF_TABLE_BITS	12
#define PF_TABLE_SZ_16	4096

#define PF_DEEP_COLOR_AWARE 1	// make sure we get 16bpc pixels; AE_Effect.h checks for this.

#include "AEConfig.h"

#ifdef AE_OS_WIN
	typedef unsigned short PixelType;
	#include <Windows.h>
#endif

#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"	// PF_GPUDeviceSuite1, GPU cmd structs
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEFX_ChannelDepthTpl.h"
#include "AEGP_SuiteHandler.h"

#include "ExtendedShadow_Strings.h"

/* Versioning information */
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

/* Slider ranges, in pixels at full resolution. */
#define	ES_LENGTH_MAX		4000
#define	ES_LENGTH_DFLT		100

/* Shadow Type popup (1-based, as AE reports popups). */
#define	ES_TYPE_DIR			1
#define	ES_TYPE_RADIAL		2
#define	ES_TYPE_INVRADIAL	3

/* Fill popup. Linear ramps along the start->end axis; Radial ramps by distance
 * from the start point (center), with |end-start| as the radius. */
#define	ES_FILL_SOLID		1
#define	ES_FILL_LINEAR		2
#define	ES_FILL_RADIAL		3

/* Parameter order. MUST match the order of PF_ADD_* in ParamsSetup. */
enum {
	ES_INPUT = 0,		// index 0 is always the input layer
	ES_TYPE,			// Directional | Radial | Inverse Radial
	ES_DIRECTION,		// angle, directional
	ES_LENGTH,			// directional length, px
	ES_SOURCE,			// radial source point (layer coords)
	ES_LENGTH_PCT,		// radial length, % of distance to source
	ES_FADE_IN,			// opacity where the shadow leaves the object, %
	ES_FADE_OUT,		// opacity at the tip, %
	ES_COLOR,			// shadow colour (solid, or gradient START)
	ES_FILL,			// Solid | Linear Gradient
	ES_COLOR2,			// gradient END colour
	ES_GSTART,			// gradient start point
	ES_GEND,			// gradient end point
	ES_TINT,			// 0 = keep object colours .. 100 = flat shadow colour
	ES_OPACITY,			// overall shadow opacity, %
	ES_THRESHOLD,		// alpha cutoff for "object", %
	ES_NUM_PARAMS
};

/* Stable IDs saved in the project file. Never reuse/renumber once shipped. */
enum {
	TYPE_DISK_ID = 1,
	DIRECTION_DISK_ID,
	LENGTH_DISK_ID,
	SOURCE_DISK_ID,
	LENGTH_PCT_DISK_ID,
	FADE_IN_DISK_ID,
	FADE_OUT_DISK_ID,
	COLOR_DISK_ID,
	FILL_DISK_ID,
	COLOR2_DISK_ID,
	GSTART_DISK_ID,
	GEND_DISK_ID,
	TINT_DISK_ID,
	OPACITY_DISK_ID,
	THRESHOLD_DISK_ID,
};

/* Per-render snapshot. Built in PreRender, consumed in SmartRender. */
typedef struct ESInfo {
	A_long		type;			// ES_TYPE_*
	PF_FpLong	angle;			// degrees (directional)
	PF_FpLong	length;			// px, downsample-scaled (directional)
	PF_FpLong	srcX, srcY;		// radial source, layer px (render res)
	PF_FpLong	lengthPct;		// 0..N as a fraction (radial)
	PF_FpLong	fadeIn, fadeOut;// 0..1
	PF_FpLong	color[3];		// straight RGB 0..1 (solid / gradient start)
	A_long		fill;			// ES_FILL_*
	PF_FpLong	color2[3];		// gradient end colour
	PF_FpLong	gsx, gsy, gex, gey;	// gradient endpoints, layer px
	PF_FpLong	tint;			// 0..1
	PF_FpLong	opacity;		// 0..1
	PF_FpLong	threshold;		// 0..1
	// Rect bookkeeping so SmartRender can map output pixels -> input pixels.
	PF_LRect	inRect;			// what we checked out
	PF_LRect	outRect;		// what we promised to fill
	A_long		pad;			// margin (px) the march must SEE around the output
} ESInfo, *ESInfoP, **ESInfoH;


extern "C" {

	DllExport
	PF_Err
	EffectMain(
		PF_Cmd			cmd,
		PF_InData		*in_data,
		PF_OutData		*out_data,
		PF_ParamDef		*params[],
		PF_LayerDef		*output,
		void			*extra);

}

#endif // EXTENDEDSHADOW_H
