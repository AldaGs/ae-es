/*
	ExtendedShadow_Strings.cpp
*/

#include "ExtendedShadow.h"

typedef struct {
	A_u_long	index;
	A_char		str[256];
} TableString;

TableString g_strs[StrID_NUMTYPES] = {
	StrID_NONE,						"",
	StrID_Name,						"Extended Shadow",
	StrID_Description,				"Long shadow: the layer's alpha silhouette projected and faded.\rPorted from python-proto/long_shadow.",
	StrID_Type_Param_Name,			"Shadow Type",
	StrID_Direction_Param_Name,		"Shadow Direction",
	StrID_Length_Param_Name,		"Shadow Length",
	StrID_Source_Param_Name,		"Shadow Source",
	StrID_LengthPct_Param_Name,		"Shadow Length %",
	StrID_FadeIn_Param_Name,		"Fade In",
	StrID_FadeOut_Param_Name,		"Fade Out",
	StrID_Color_Param_Name,			"Shadow Color",
	StrID_Fill_Param_Name,			"Fill",
	StrID_Color2_Param_Name,		"Gradient End Color",
	StrID_GStart_Param_Name,		"Gradient Start",
	StrID_GEnd_Param_Name,			"Gradient End",
	StrID_Tint_Param_Name,			"Tint Amount",
	StrID_Opacity_Param_Name,		"Opacity",
	StrID_Threshold_Param_Name,		"Alpha Threshold",
	StrID_Type_Choices,				"Directional|Radial|Inverse Radial",
	StrID_Fill_Choices,				"Solid|Linear Gradient",
};

char *GetStringPtr(int strNum)
{
	return g_strs[strNum].str;
}
