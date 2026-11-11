
// CHECKLICENSE
// much stuff from https://github.com/mkkellogg/GaussianSplats3D/blob/main/LICENSE
// (MIT license)

#define CUB_DISABLE_BF16_SUPPORT

#define GLM_FORCE_CUDA
#define CUDA_VERSION 12000

namespace std{
	using size_t = ::size_t;
};

using namespace std;

#include "./libs/glm/glm/glm.hpp"
#include "./libs/glm/glm/gtc/matrix_transform.hpp"
// #include "./libs/glm/glm/gtc/matrix_access.hpp"
#include "./libs/glm/glm/gtx/transform.hpp"
#include "./libs/glm/glm/gtc/quaternion.hpp"
#include "./libs/glm/glm/gtx/matrix_decompose.hpp"

#include "utils.cuh"
#include "HostDeviceInterface.h"
#include "math.cuh"

namespace cg = cooperative_groups;

using glm::ivec2;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::uvec3;
using glm::uvec4;
using glm::mat3;
using glm::mat4;
using glm::quat;
using glm::dot;
using glm::transpose;
using glm::inverse;

constexpr int VIEWMODE_DESKTOP = 0;
constexpr int VIEWMODE_DESKTOP_VR = 1;
constexpr int VIEWMODE_IMMERSIVE_VR = 2;
constexpr uint32_t BACKGROUND_COLOR = 0xff000000;
constexpr uint64_t DEFAULT_PIXEL = (uint64_t(Infinity) << 32) | BACKGROUND_COLOR;

__device__ uint64_t g_uclosest;

__device__ double g_x;
__device__ double g_y;
__device__ double g_z;
__device__ uint32_t g_numSelected;
__device__ bool dbg_bool;
__device__ uint32_t g_counter;

constexpr int dbgtile_x = 60;
constexpr int dbgtile_y = 50;

void setBit(void* buffer, uint32_t bitIndex){
	uint32_t wordIndex = bitIndex / 32;
	uint32_t localBitIndex = bitIndex % 32;

	uint32_t* bu32 = (uint32_t*)buffer;
	uint32_t mask = 1 << localBitIndex;

	atomicOr(&bu32[wordIndex], mask);
}

void clearBit(void* buffer, uint32_t bitIndex){
	uint32_t wordIndex = bitIndex / 32;
	uint32_t localBitIndex = bitIndex % 32;

	uint32_t* bu32 = (uint32_t*)buffer;
	uint32_t mask = ~(1 << localBitIndex);

	atomicAnd(&bu32[wordIndex], mask);
}

bool getBit(void* buffer, uint32_t bitIndex){
	uint32_t wordIndex = bitIndex / 32;
	uint32_t localBitIndex = bitIndex % 32;

	uint32_t* bu32 = (uint32_t*)buffer;

	uint32_t word = bu32[wordIndex];
	bool isSet = ((word >> localBitIndex) & 1) == 1;

	return isSet;
}

vec3 getRayDir(vec2 pixelCoord, RenderTarget target){

	// prepare rays
	vec3 origin = vec3(inverse(target.view) * vec4(0.0, 0.0, 0.0, 1.0));

	vec4 dir_00_projspace = vec4{-1.0f, -1.0f, 1.0f, 1.0f};
	vec4 dir_01_projspace = vec4{-1.0f,  1.0f, 1.0f, 1.0f};
	vec4 dir_10_projspace = vec4{ 1.0f, -1.0f, 1.0f, 1.0f};
	vec4 dir_11_projspace = vec4{ 1.0f,  1.0f, 1.0f, 1.0f};

	float right = 1.0f / target.proj[0][0];
	float up = 1.0f / target.proj[1][1];
	vec4 dir_00_worldspace = inverse(target.view) * vec4(-right, -up, -1.0f, 1.0f);
	vec4 dir_01_worldspace = inverse(target.view) * vec4(-right,  up, -1.0f, 1.0f);
	vec4 dir_10_worldspace = inverse(target.view) * vec4( right, -up, -1.0f, 1.0f);
	vec4 dir_11_worldspace = inverse(target.view) * vec4( right,  up, -1.0f, 1.0f);

	auto getRayDir = [&](float u, float v){
		float A_00 = (1.0f - u) * (1.0f - v);
		float A_01 = (1.0f - u) *         v;
		float A_10 =         u  * (1.0f - v);
		float A_11 =         u  *         v;

		vec3 dir = (
			A_00 * dir_00_worldspace +
			A_01 * dir_01_worldspace +
			A_10 * dir_10_worldspace +
			A_11 * dir_11_worldspace - vec4(origin, 1.0));
		dir = normalize(dir);

		return dir;
	};

	// auto mouse = args.mouseEvents;
	float u = pixelCoord.x / target.width;
	float v = pixelCoord.y / target.height;
	vec3 dir = getRayDir(u, v);

	return dir;
}



// Select all splats in model
extern "C" __global__
void kernel_select(CommonLaunchArgs args, GaussianData model, uint32_t* changedmask){
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index >= model.count) return;

	uint32_t flags = model.flags[index];
	uint32_t newFlags = flags | FLAGS_SELECTED;

	if(newFlags != flags){
		model.flags[index] = newFlags;

		if(changedmask){
			setBit(changedmask, index);
		}
	}
}

// Deselect all splats in model
extern "C" __global__
void kernel_deselect(CommonLaunchArgs args, GaussianData model, uint32_t* changedmask){
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index >= model.count) return;

	uint32_t flags = model.flags[index];
	uint32_t newFlags = flags;
	newFlags = newFlags & (~FLAGS_HIGHLIGHTED);
	newFlags = newFlags & (~FLAGS_HIGHLIGHTED_NEGATIVE);
	newFlags = newFlags & (~FLAGS_SELECTED);

	if(newFlags != flags){
		model.flags[index] = newFlags;

		if(changedmask){
			setBit(changedmask, index);
		}
	}
}

extern "C" __global__
void kernel_invert_selection(CommonLaunchArgs args, GaussianData model){
	int index = cg::this_grid().thread_rank();

	if(index >= model.count) return;

	uint32_t selectionFlag = model.flags[index] & FLAGS_SELECTED;
	selectionFlag = !selectionFlag;

	uint32_t mask = ~1;
	model.flags[index] = (model.flags[index] & mask) | selectionFlag;
}

extern "C" __global__
void kernel_delete_selection(GaussianData model, uint32_t* deletedMask){
	int index = cg::this_grid().thread_rank();

	if(index >= model.count) return;

	uint32_t flags = model.flags[index];

	bool isSelected = flags & FLAGS_SELECTED;
	bool isDeleted = flags & FLAGS_DELETED;

	if(isSelected && !isDeleted){

		flags = flags | FLAGS_DELETED;
		flags = flags & ~FLAGS_SELECTED;

		model.flags[index] = flags;

		if(deletedMask){
			setBit(deletedMask, index);
		}
	}
}

extern "C" __global__
void kernel_undelete_selection(GaussianData model, uint32_t* deletionmask){
	int index = cg::this_grid().thread_rank();

	if(index >= model.count) return;


	bool bit = getBit(deletionmask, index);

	if(bit){

		uint32_t flags = model.flags[index];

		// remove deletion flag, set selection flag
		flags = flags & ~FLAGS_DELETED;
		flags = flags | FLAGS_SELECTED;

		model.flags[index] = flags;
	}
}

extern "C" __global__
void kernel_apply_colorCorrection(
	CommonLaunchArgs args, GaussianData model, ColorCorrection colorCorrection
){
	uint32_t index = cg::this_grid().thread_rank();
	if(index >= model.count) return;

	Color C = model.color[index];
	vec4 color = C.normalized();
	vec4 corrected = applyColorCorrection(color, colorCorrection);
	
	Color C_corrected = Color::fromNormalized(corrected);
	model.color[index] = C_corrected;
}

// Applies <transform> to all splats or <onlySelected> splats of the model. 
extern "C" __global__
void kernel_apply_transformation(
	CommonLaunchArgs args, GaussianData model, mat4 transform, 
	uint32_t first, uint32_t count,
	bool onlySelected
){

	uint32_t index = cg::this_grid().thread_rank();
	uint32_t splatIndex = index + first;

	if(splatIndex >= model.count) return;
	if(splatIndex >= first + count) return;

	if(onlySelected){
		bool isSelected = model.flags[splatIndex] & FLAGS_SELECTED;

		if(!isSelected) return;
	}

	vec3 d_scale;
	quat d_rotation;
	vec3 d_translation;
	vec3 d_skew;
	vec4 d_perspective;
	glm::decompose(transform, d_scale, d_rotation, d_translation, d_skew, d_perspective);
	d_rotation = glm::conjugate(d_rotation);

	{ // TRANSFORM POSITION
		vec3 pos = model.position[splatIndex];
		vec3 newPos = vec3(transform * vec4(pos, 1.0f));
		model.position[splatIndex] = newPos;
	}

	{ // ROTATION
		quat q = quat(
			model.quaternion[splatIndex].x,
			model.quaternion[splatIndex].y,
			model.quaternion[splatIndex].z,
			model.quaternion[splatIndex].w
		);

		q = d_rotation * q;
		float l = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
		l = 1.0f;

		model.quaternion[splatIndex].x = q.w / l;
		model.quaternion[splatIndex].y = q.x / l;
		model.quaternion[splatIndex].z = q.y / l;
		model.quaternion[splatIndex].w = q.z / l;
	}

	if(model.shDegree > 0)
	{ // SPHERICAL HARMONICS
		mat3 rotation = glm::mat3_cast(d_rotation);
		
		int64_t offset = splatIndex * model.numSHCoefficients;
		vec3* shs = (vec3*)(model.sphericalHarmonics + offset);
		vec3 shs_transformed[15];
		rotateSH(model.shDegree, shs, shs_transformed, rotation);

		int numCoefficients = shDegreeToNumCoefficients(model.shDegree);
		memcpy(shs, shs_transformed, (numCoefficients - 1) * sizeof(vec3));
	}

	{
		vec3 scale = model.scale[splatIndex];
		scale = scale * d_scale;
		model.scale[splatIndex] = scale;
	}
}

__device__ uint32_t g_copyCounter;
extern "C" __global__
void kernel_copy_model(CommonLaunchArgs args, GaussianData source, GaussianData target, bool ignoreDeleted){

	auto grid = cg::this_grid();

	g_copyCounter = 0;

	// On host side, we prepared a target with memory for target.count bytes.
	// target.count may be smaller than source.count if we ignore deleted splats.
	// For safety, we explicitly check targetsplat[g_copyCounter] against target.count

	grid.sync();

	process(source.count, [&](int sourceIndex){

		if(ignoreDeleted){
			bool accept = (source.flags[sourceIndex] & FLAGS_DELETED) == 0;
			if(!accept) return;
		}

		uint32_t targetIndex = atomicAdd(&g_copyCounter, 1);

		if(targetIndex >= target.count){
			// ERROR
			if(targetIndex == target.count){
				printf("ERROR: Tried to copy, but allocated target model is too small\n");
			}
			return;
		}

		// now copy attributes
		target.position[targetIndex]   = source.position[sourceIndex];
		target.scale[targetIndex]      = source.scale[sourceIndex];
		target.quaternion[targetIndex] = source.quaternion[sourceIndex];
		target.color[targetIndex]      = source.color[sourceIndex];
		target.flags[targetIndex]      = source.flags[sourceIndex];
		// target.depth[targetIndex] = source.depth[sourceIndex];
	});

	grid.sync();

	if(grid.thread_rank() == 0){
		if(g_copyCounter != target.count){
			printf("ERROR: Number of (attempted) copied splats is different than number of allocated splats in copy.\n");
			printf("expected copies: %d, (attempted) copies: %d \n", target.count, g_copyCounter);
		}
	}

}

extern "C" __global__
void kernel_rectselect(
	CommonLaunchArgs args, 
	RenderTarget target,
	GaussianData model,
	RectSelect rect,
	bool apply,
	uint32_t* splatmask
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	uint32_t index = grid.thread_rank();

	if(index >= model.count) return;

	// handle selection / deselection
	mat4 transform = target.proj * target.view;

	uint32_t flags = model.flags[index];
	uint32_t newFlags = flags;
	vec3 pos = model.position[index];
	vec3 worldPos = vec3(model.transform * vec4(pos, 1.0f));

	vec4 ndc = transform * vec4(worldPos, 1.0f);
	vec2 imgCoords = vec2(
		(0.5f * (ndc.x / ndc.w) + 0.5f) * target.width,
		(0.5f * (ndc.y / ndc.w) + 0.5f) * target.height
	);

	bool isDeleted = model.flags[index] & FLAGS_DELETED;
	if(isDeleted) return;

	auto intersectsPoint = [&](vec2 point, vec2 min, vec2 max) -> bool {
		bool insideX = point.x > min.x && point.x < max.x;
		bool insideY = point.y > min.y && point.y < max.y;
		// insideY = point.y < max.y;

		// return insideY;
		return insideX && insideY;
	};

	auto intersectsCenter = [&]() -> bool {
		bool intersectsX = imgCoords.x > min(rect.start.x, rect.end.x) && imgCoords.x < max(rect.start.x, rect.end.x);
		bool intersectsY = imgCoords.y > min(rect.start.y, rect.end.y) && imgCoords.y < max(rect.start.y, rect.end.y);
		bool intersects = intersectsX && intersectsY;

		return intersects;
	};

	auto intersectsBorder = [&](int index) -> bool {
		mat4 world = model.transform;
		mat4 view = target.view;
		mat4 proj = target.proj;
		mat4 worldView = view * world;
		mat4 worldViewProj = proj * view * world;

		vec3 splatPos = model.position[index];
		vec4 worldPos = world * vec4(splatPos, 1.0f);
		vec4 viewPos = view * worldPos;
		vec4 ndc = proj * viewPos;

		ndc.x = ndc.x / ndc.w;
		ndc.y = ndc.y / ndc.w;
		ndc.z = ndc.z / ndc.w;

		if(ndc.w <= 0.0f) return false;
		if(ndc.x < -1.1f || ndc.x >  1.1f) return false;
		if(ndc.y < -1.1f || ndc.y >  1.1f) return false;

		vec4 quat = model.quaternion[index];
		mat3 rotation = glm::mat3_cast(glm::quat(quat.x, quat.y, quat.z, quat.w));

		mat3 scale = mat3(1.0f);
		scale[0][0] = model.scale[index].x;
		scale[1][1] = model.scale[index].y;
		scale[2][2] = model.scale[index].z;

		mat3 cov3D = rotation * scale * scale * transpose(rotation);

		vec2 focal = vec2(
			proj[0][0] * target.width * 0.5f,
			proj[1][1] * target.height * 0.5f
		);
		float s = 1.0f / (viewPos.z * viewPos.z);
		mat3 J = mat3(
			focal.x / viewPos.z   , 0.0f                  , -(focal.x * viewPos.x) * s,
			0.0f                  , focal.y / viewPos.z   , -(focal.y * viewPos.y) * s,
			0.0f                  , 0.0f                  , 0.0f
		);

		mat3 W = transpose(mat3(worldView));
		mat3 T = W * J;

		mat3 cov2Dm = transpose(T) * cov3D * T;
		cov2Dm[0][0] += 0.2f;
		cov2Dm[1][1] += 0.2f;

		vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

		float a = cov2Dv.x;
		float d = cov2Dv.z;
		float b = cov2Dv.y;
		float D = a * d - b * b;
		float trace = a + d;
		float traceOver2 = 0.5f * trace;
		float term2 = sqrt(max(0.05f, traceOver2 * traceOver2 - D));
		float eigenValue1 = traceOver2 + term2;
		float eigenValue2 = traceOver2 - term2;
		eigenValue1 = max(eigenValue1, 0.0f);
		eigenValue2 = max(eigenValue2, 0.0f);

		vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
		vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

		float splatScale = args.uniforms.splatSize;
		const float sqrt8 = sqrt(8.0f);

		vec2 basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), MAX_SCREENSPACE_SPLATSIZE);
		vec2 basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), MAX_SCREENSPACE_SPLATSIZE);

		vec2 pixelCoord = {
			((ndc.x) * 0.5f + 0.5f) * target.width,
			((ndc.y) * 0.5f + 0.5f) * target.height
		};

		vec2 lower = {
			min(rect.start.x, rect.end.x),
			min(rect.start.y, rect.end.y),
		};
		vec2 upper = {
			max(rect.start.x, rect.end.x),
			max(rect.start.y, rect.end.y),
		};
		
		int numSegments = (length(basisVector1) + length(basisVector2)) / 10;
		
		// assert safe limits
		numSegments = clamp(numSegments, 0, 100);

		for(int i = 0; i < numSegments; i++){
			float u = 2.0f * 3.1415f * float(i) / float(numSegments);

			float wx = cos(u);
			float wy = sin(u);

			vec2 point = pixelCoord + basisVector1 * wx + basisVector2 * wy;

			if(intersectsPoint(point, lower, upper)) return true;
		}

		if(intersectsPoint(pixelCoord, lower, upper)) return true;

		return false;

	};

	if(rect.active && rect.startpos_specified){

		bool intersects = false;
		if(args.brush.intersectionmode == BRUSH_INTERSECTION_CENTER){
			intersects = intersectsCenter();
		}else if(args.brush.intersectionmode == BRUSH_INTERSECTION_BORDER){
			intersects = intersectsBorder(index);
		}

		if(intersects){
			if(apply){
				if(rect.unselecting){
					newFlags = newFlags & (~FLAGS_SELECTED);
				}else{
					newFlags = newFlags | FLAGS_SELECTED;
				}

			}else if(rect.unselecting){
				newFlags = newFlags | FLAGS_HIGHLIGHTED_NEGATIVE;
			}else{
				newFlags = newFlags | FLAGS_HIGHLIGHTED;
			}
		}else{
			newFlags = newFlags & (~FLAGS_HIGHLIGHTED);
			newFlags = newFlags & (~FLAGS_HIGHLIGHTED_NEGATIVE);
		}
	}

	if(apply){
		newFlags = newFlags & (~FLAGS_HIGHLIGHTED);
		newFlags = newFlags & (~FLAGS_HIGHLIGHTED_NEGATIVE);
	}

	bool wasSelected = flags & FLAGS_SELECTED;
	bool isSelected = newFlags & FLAGS_SELECTED;

	if(model.flags[index] != newFlags){
		model.flags[index] = newFlags;
	}
}

extern "C" __global__
void kernel_select_vr(
	CommonLaunchArgs args, 
	GaussianData model,
	bool isTriggerPressed,
	mat4 controllerPose,
	uint32_t* changedmask
){
	auto grid = cg::this_grid();
	uint32_t index = grid.thread_rank();

	if(index >= model.count) return;

	const uint32_t flags = model.flags[index];
	uint32_t newFlags = flags;

	mat4 flip = mat4(
		1.0,  0.0, 0.0, 0.0,
		0.0,  0.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 0.0,
		0.0,  0.0, 0.0, 1.0
	);

	uint32_t intersectionMode = BRUSH_INTERSECTION_BORDER;

	vec3 d_scale;
	quat d_rotation;
	vec3 d_translation;
	vec3 d_skew;
	vec4 d_perspective;
	glm::decompose(model.transform, d_scale, d_rotation, d_translation, d_skew, d_perspective);
	d_rotation = glm::conjugate(d_rotation);

	mat4 mTranslate = translate(vec3{0.0f, 0.1f, 0.0f});
	mat4 mScale = scale(vec3{0.1f, 0.1f, 0.1f});
	mat4 mRot = glm::rotate(-140.0f * 3.1415f / 180.0f, vec3{ 1.0f, 0.0f, 0.0f });
	mat4 transform = mat4(flip * controllerPose) * mRot * mTranslate * mScale;

	vec3 controllerPos = vec3(transform * vec4{0.0f, 0.0f, 0.0f, 1.0f});
	float brushRadius = 0.1f;

	
	if(intersectionMode == BRUSH_INTERSECTION_CENTER){

		vec3 pos = model.position[index];
		vec3 worldPos = vec3(model.transform * vec4(pos, 1.0f));

		bool isInBrushRadius = length(controllerPos - worldPos) < brushRadius;

		if(isInBrushRadius){
			newFlags = newFlags | FLAGS_HIGHLIGHTED;
		}else{
			newFlags = newFlags & (~FLAGS_HIGHLIGHTED);
		}

		if(isTriggerPressed && isInBrushRadius && args.brush.mode == BRUSHMODE::SELECT){
			newFlags = newFlags | FLAGS_SELECTED;
		}else if(isTriggerPressed && isInBrushRadius && args.brush.mode == BRUSHMODE::ERASE){
			newFlags = newFlags | FLAGS_DELETED;
		}else if(isTriggerPressed && isInBrushRadius && args.brush.mode == BRUSHMODE::REMOVE_FLAGS){
			newFlags = newFlags & (~0b1001);
		}

	}else if(intersectionMode == BRUSH_INTERSECTION_BORDER){

		vec3 pos = model.position[index];
		vec3 worldPos = vec3(model.transform * vec4(pos, 1.0f));

		vec4 quat = model.quaternion[index];
		mat3 rotation = glm::mat3_cast(glm::quat(quat.x, quat.y, quat.z, quat.w));
		mat4 rot4 = mat4(rotation);
		rot4[3][3] = 1.0f;
		vec3 scale = model.scale[index];

		{ // apply the splat model's transformation
			glm::quat q = glm::quat(quat.x, quat.y, quat.z, quat.w);
			q = d_rotation * q;

			rotation = glm::mat3_cast(q);
			rot4 = mat4(rotation);
			rot4[3][3] = 1.0f;

			scale = scale * d_scale;
		}

		vec3 dx = /** world * **/ rot4 * vec4{scale.x, 0.0f, 0.0f, 0.0f};
		vec3 dy = /** world * **/ rot4 * vec4{0.0f, scale.y, 0.0f, 0.0f};
		vec3 dz = /** world * **/ rot4 * vec4{0.0f, 0.0f, scale.z, 0.0f};

		dx = {abs(dx.x), abs(dx.y), abs(dx.z)};
		dy = {abs(dy.x), abs(dy.y), abs(dy.z)};
		dz = {abs(dz.x), abs(dz.y), abs(dz.z)};

		vec3 size = {
			max(max(dx.x, dy.x), dz.x),
			max(max(dx.y, dy.y), dz.y),
			max(max(dx.z, dy.z), dz.z),
		};

		float d = length(controllerPos - worldPos);

		auto intersect = [](vec3 point, vec3 splatPos, vec3 basisVector1, vec3 basisVector2, vec3 basisVector3) -> bool{

			float a = length(basisVector1);
			float b = length(basisVector2);
			float c = length(basisVector3);

			vec3 pFrag = point - splatPos;
			float sA = dot(normalize(basisVector1), pFrag) / a;
			float sB = dot(normalize(basisVector2), pFrag) / b;
			float sC = dot(normalize(basisVector3), pFrag) / c;

			float w = sqrt(sA * sA + sB * sB + sC * sC);

			return w < 1.0f;
		};
		
		vec3 basisVector1 = vec3{1.0f, 0.0f, 0.0f} * size.x * 2.8f;
		vec3 basisVector2 = vec3{0.0f, 1.0f, 0.0f} * size.y * 2.8f;
		vec3 basisVector3 = vec3{0.0f, 0.0f, 1.0f} * size.z * 2.8f;

		// float diag = radius * 2.0f;
		float a = length(basisVector1);
		float b = length(basisVector2);
		float c = length(basisVector3);
		float a_outer = (a + brushRadius);
		float b_outer = (b + brushRadius);
		float c_outer = (c + brushRadius);

		bool intersects = intersect(controllerPos, worldPos, 
			a_outer * basisVector1 / a, 
			b_outer * basisVector2 / b, 
			c_outer * basisVector3 / c
		);

		if(intersects){
			newFlags = newFlags | FLAGS_HIGHLIGHTED;
		}else{
			newFlags = newFlags & (~FLAGS_HIGHLIGHTED);
		}

		if(isTriggerPressed && intersects && args.brush.mode == BRUSHMODE::SELECT){
			newFlags = newFlags | FLAGS_SELECTED;
		}else if(isTriggerPressed && intersects && args.brush.mode == BRUSHMODE::ERASE){
			newFlags = newFlags | FLAGS_DELETED;
		}else if(isTriggerPressed && intersects && args.brush.mode == BRUSHMODE::REMOVE_FLAGS){
			newFlags = newFlags & (~0b1001);
		}
	}

	if(newFlags != flags){
		model.flags[index] = newFlags;
		
		bool wasSelected = flags & FLAGS_SELECTED;
		bool isSelected = newFlags & FLAGS_SELECTED;
		bool wasDeleted = flags & FLAGS_DELETED;
		bool isDeleted = newFlags & FLAGS_DELETED;

		if(changedmask && (wasSelected != isSelected) | (wasDeleted != isDeleted)){
			setBit(changedmask, index);
		}
	}
}

extern "C" __global__
void kernel_clear_highlighting(CommonLaunchArgs args, GaussianData model){
	auto grid = cg::this_grid();
	uint32_t index = grid.thread_rank();

	if(index >= model.count) return;

	uint32_t flags = model.flags[index];

	flags = flags & ~FLAGS_HIGHLIGHTED;
	flags = flags & ~FLAGS_HIGHLIGHTED_NEGATIVE;

	model.flags[index] = flags;
}



extern "C" __global__
void kernel_select_sphere(
	CommonLaunchArgs args, 
	GaussianData model,
	vec3 spherePos,
	float radius,
	uint32_t* splatmask
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int index = grid.thread_rank();

	if(index >= model.count) return;

	uint32_t intersectionMode = BRUSH_INTERSECTION_CENTER;

	// spherePos = {0.0f, 0.0f, 0.0f};
	// radius = 1.0f;

	if(intersectionMode == BRUSH_INTERSECTION_CENTER){
		vec3 pos = model.position[index];
		pos = vec3(model.transform * vec4{pos.x, pos.y, pos.z, 1.0f});

		float d = length(spherePos - pos);

		uint32_t flags = model.flags[index];

		if(d < radius){
			model.flags[index] = flags | FLAGS_HIGHLIGHTED;

			if(args.mouseEvents.isLeftDown){
				bool alreadySelected = flags & FLAGS_SELECTED;

				if(!alreadySelected){
					model.flags[index] = flags | FLAGS_SELECTED;

					if(splatmask){
						setBit(splatmask, index);
					}
				}
			}
		}else{
			uint32_t newFlags = flags & ~FLAGS_HIGHLIGHTED;

			if(flags != newFlags){
				model.flags[index] = newFlags;
			}
		}
	}else if(intersectionMode == BRUSH_INTERSECTION_BORDER){
		vec3 pos = model.position[index];
		vec3 worldPos = vec3(model.transform * vec4(pos, 1.0f));

		vec4 quat = model.quaternion[index];
		mat3 rotation = glm::mat3_cast(glm::quat(quat.x, quat.y, quat.z, quat.w));
		// mat4 rot4;
		mat4 rot4 = mat4(rotation);
		rot4[3][3] = 1.0f;
		vec3 scale = model.scale[index];

		{ // apply the splat model's transformation

			vec3 d_scale;
			glm::quat d_rotation;
			vec3 d_translation;
			vec3 d_skew;
			vec4 d_perspective;
			glm::decompose(model.transform, d_scale, d_rotation, d_translation, d_skew, d_perspective);
			d_rotation = glm::conjugate(d_rotation);

			// glm::quat q = glm::quat(quat.x, quat.y, quat.z, quat.w);
			// q = d_rotation * q;

			// rotation = glm::mat3_cast(q);
			// rot4 = mat4(rotation);
			// rot4[3][3] = 1.0f;

			scale = scale * d_scale;
		}

		vec3 dx = /** world * **/ rot4 * vec4{scale.x, 0.0f, 0.0f, 0.0f};
		vec3 dy = /** world * **/ rot4 * vec4{0.0f, scale.y, 0.0f, 0.0f};
		vec3 dz = /** world * **/ rot4 * vec4{0.0f, 0.0f, scale.z, 0.0f};

		dx = {abs(dx.x), abs(dx.y), abs(dx.z)};
		dy = {abs(dy.x), abs(dy.y), abs(dy.z)};
		dz = {abs(dz.x), abs(dz.y), abs(dz.z)};

		vec3 size = {
			max(max(dx.x, dy.x), dz.x),
			max(max(dx.y, dy.y), dz.y),
			max(max(dx.z, dy.z), dz.z),
		};

		float d = length(spherePos - worldPos);

		auto intersect = [](vec3 point, vec3 splatPos, vec3 basisVector1, vec3 basisVector2, vec3 basisVector3) -> bool{

			float a = length(basisVector1);
			float b = length(basisVector2);
			float c = length(basisVector3);

			vec3 pFrag = point - splatPos;
			float sA = dot(normalize(basisVector1), pFrag) / a;
			float sB = dot(normalize(basisVector2), pFrag) / b;
			float sC = dot(normalize(basisVector3), pFrag) / c;

			float w = sqrt(sA * sA + sB * sB + sC * sC);

			return w < 1.0f;
		};
		
		vec3 basisVector1 = vec3{1.0f, 0.0f, 0.0f} * size.x * 2.8f;
		vec3 basisVector2 = vec3{0.0f, 1.0f, 0.0f} * size.y * 2.8f;
		vec3 basisVector3 = vec3{0.0f, 0.0f, 1.0f} * size.z * 2.8f;

		// float diag = radius * 2.0f;
		float a = length(basisVector1);
		float b = length(basisVector2);
		float c = length(basisVector3);
		float a_outer = (a + radius);
		float b_outer = (b + radius);
		float c_outer = (c + radius);

		bool intersects = intersect(spherePos, worldPos, 
			a_outer * basisVector1 / a, 
			b_outer * basisVector2 / b, 
			c_outer * basisVector3 / c
		);

		uint32_t flags = model.flags[index];

		if(intersects){
			model.flags[index] = model.flags[index] | FLAGS_HIGHLIGHTED;

			if(args.mouseEvents.isLeftDown){
				bool alreadySelected = model.flags[index] & FLAGS_SELECTED;

				if(!alreadySelected){
					model.flags[index] = model.flags[index] | FLAGS_SELECTED;

					if(splatmask){
						setBit(splatmask, index);
					}
				}
			}
		}else{
			uint32_t newFlags = flags & ~FLAGS_HIGHLIGHTED;

			if(flags != newFlags){
				model.flags[index] = newFlags;
			}
		}
	}
}

extern "C" __global__
void kernel_paint_sphere(
	CommonLaunchArgs args, 
	GaussianData model,
	Color* stashedColors,
	vec3 spherePos,
	float radius,
	vec4 paintColor,
	int numTicks
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int index = grid.thread_rank();

	if(index >= model.count) return;

	bool isDeleted = model.flags[index] & FLAGS_DELETED;
	if(isDeleted) return;

	vec3 pos = model.position[index];
	pos = vec3(model.transform * vec4{pos.x, pos.y, pos.z, 1.0f});

	float d = length(spherePos - pos);

	if(d < radius){

		if(args.uniforms.brushColorMode == BRUSHCOLORMODE_NORMAL){
			float w_dist = clamp((radius - d) / radius, 0.0f, 1.0f);
			float w_opacity = paintColor.a;
			float w = w_dist * w_opacity;

			float splatOpacity = stashedColors[index].normalized().a;
			// Current color, modified by previous brush steps
			vec3 original = stashedColors[index].normalized();
			vec3 current = model.color[index].normalized();
			vec3 newCurrent = (1.0f - w) * current + w * vec3{paintColor};
			vec3 target = (1.0f - w) * original + w * vec3{paintColor};
			vec3 max = (1.0f - w_opacity) * original + w_opacity * vec3{paintColor};

			vec3 dir = normalize(target - original);
			float u_target = dot(dir, target - original);
			float u_current = dot(dir, current - original);
			float u_newCurrent = dot(dir, newCurrent - original);
			float u_max = dot(dir, max - original);

			if(u_newCurrent > u_max){
				model.color[index] = Color::fromNormalized(vec4{max, splatOpacity});
			}else{
				model.color[index] = Color::fromNormalized(vec4{newCurrent, splatOpacity});
			}
		}else if(args.uniforms.brushColorMode == BRUSHCOLORMODE_HUE_SATURATION){
			float splatOpacity = stashedColors[index].normalized().a;
			vec3 csplat = model.color[index].normalized();

			vec3 hsl_brush = rgbToHSL(paintColor.r, paintColor.g, paintColor.b);
			vec3 hsl_splat = rgbToHSL(csplat.r, csplat.g, csplat.b);
			
			hsl_splat.r = hsl_brush.r;
			hsl_splat.g = hsl_brush.g;
			vec3 result = hslToRgb(hsl_splat.r, hsl_splat.g, hsl_splat.b);

			model.color[index] = Color::fromNormalized(vec4{result, splatOpacity});
		}

	}
}

extern "C" __global__
void kernel_swap_colors(
	Color* a,
	Color* b,
	uint32_t numSplats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int index = grid.thread_rank();

	if(index >= numSplats) return;

	Color tmp = a[index];
	a[index] = b[index];
	b[index] = tmp;
}

extern "C" __global__
void kernel_countChangedColors(
	Color* original,
	Color* modified,
	uint32_t numSplats,
	uint32_t* counter
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int index = grid.thread_rank();

	if(index >= numSplats) return;

	Color a = original[index];
	Color b = modified[index];
	bool isDifferent = a.r != b.r || a.g != b.g || a.b != b.b || a.a != b.a;
	
	if(isDifferent){
		atomicAdd(counter, 1);
	}
}


extern "C" __global__
void kernel_create_color_diff(
	Color* original,
	Color* modified,
	uint32_t numSplats,
	uint32_t* diff_indices,
	Color* diff_colors,
	uint32_t* counter
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int index = grid.thread_rank();

	if(index >= numSplats) return;

	Color a = original[index];
	Color b = modified[index];
	bool isDifferent = a.r != b.r || a.g != b.g || a.b != b.b || a.a != b.a;
	
	if(isDifferent){
		uint32_t targetIndex = atomicAdd(counter, 1);

		diff_indices[targetIndex] = index;
		diff_colors[targetIndex] = a;
	}
}

extern "C" __global__
void kernel_color_diff_swap(
	uint32_t* diff_indices,
	Color* diff_colors,
	Color* colors,
	uint32_t numDiffs
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int index = grid.thread_rank();
	if(index >= numDiffs) return;

	uint32_t splatIndex = diff_indices[index];

	Color tmp = colors[splatIndex];
	colors[splatIndex] = diff_colors[index];
	diff_colors[index] = tmp;
}


extern "C" __global__
void kernel_setup_splatmask(
	GaussianData model,
	uint32_t* splatmask
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int splatIndex = grid.thread_rank();

	if(splatIndex >= model.count) return;

	uint32_t flags = model.flags[splatIndex];
	bool isSelected = flags & FLAGS_SELECTED;

	if(isSelected){
		uint32_t maskIndex = splatIndex / 32;
		uint32_t bitIndex = splatIndex % 32;
		uint32_t mask = 1 << bitIndex;

		atomicOr(&splatmask[maskIndex], mask);
	}
}

extern "C" __global__
void kernel_get_selectionmask(
	GaussianData model,
	uint32_t* mask
){
	auto grid = cg::this_grid();
	int splatIndex = grid.thread_rank();

	if(splatIndex >= model.count) return;

	uint32_t flags = model.flags[splatIndex];
	bool isSelected = flags & FLAGS_SELECTED;
	
	if(isSelected){
		setBit(mask, splatIndex);
	}else{
		clearBit(mask, splatIndex);
	}
}

// extern "C" __global__
// void kernel_invertBits(
// 	uint32_t* mask, uint32_t numU32
// ){
// 	auto grid = cg::this_grid();
// 	int index = grid.thread_rank();

// 	if(index >= numU32) return;

// 	mask[index] = ~mask[index];
// }

extern "C" __global__
void kernel_set_selection(
	GaussianData model,
	uint32_t* selectionMask,
	bool inverted
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int splatIndex = grid.thread_rank();

	if(splatIndex >= model.count) return;

	uint32_t flags = model.flags[splatIndex];

	flags = flags & ~FLAGS_SELECTED;

	bool isSet = getBit(selectionMask, splatIndex);

	if(inverted){
		isSet = !isSet;
	}

	if(isSet){
		flags = flags | FLAGS_SELECTED;
	}

	model.flags[splatIndex] = flags;
}

extern "C" __global__
void kernel_swap_selection(
	GaussianData model,
	uint32_t* selectionMask
){
	auto grid = cg::this_grid();
	int splatIndex = grid.thread_rank();
	if(splatIndex >= model.count) return;

	uint32_t flags = model.flags[splatIndex];

	bool selected_model = flags & FLAGS_SELECTED;
	bool selected_mask = getBit(selectionMask, splatIndex);

	// set model's flags
	uint32_t newFlags = flags;
	newFlags = newFlags & ~FLAGS_SELECTED;
	if(selected_mask){
		newFlags = newFlags | FLAGS_SELECTED;
	}
	model.flags[splatIndex] = newFlags;
	
	// set mask's bit
	if(selected_model){
		setBit(selectionMask, splatIndex);
	}else{
		clearBit(selectionMask, splatIndex);
	}
}

extern "C" __global__
void kernel_select_masked(
	GaussianData model,
	uint32_t* selectionMask
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int splatIndex = grid.thread_rank();

	if(splatIndex >= model.count) return;

	// uint32_t maskIndex = splatIndex / 32;
	// uint32_t bitIndex = splatIndex % 32;
	// uint32_t mask = 1 << bitIndex;

	// bool isSet = (selectionMask[maskIndex] & mask ) != 0;
	bool isSet = getBit(selectionMask, splatIndex);

	if(isSet){
		uint32_t flags = model.flags[splatIndex];

		flags = flags | FLAGS_SELECTED;

		model.flags[splatIndex] = flags;
	}
}

extern "C" __global__
void kernel_delete_masked(
	GaussianData model,
	uint32_t* selectionMask
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int splatIndex = grid.thread_rank();

	if(splatIndex >= model.count) return;

	uint32_t maskIndex = splatIndex / 32;
	uint32_t bitIndex = splatIndex % 32;
	uint32_t mask = 1 << bitIndex;

	bool isSet = (selectionMask[maskIndex] & mask ) != 0;

	if(isSet){
		uint32_t flags = model.flags[splatIndex];

		flags = flags | FLAGS_DELETED;

		model.flags[splatIndex] = flags;
	}
}

extern "C" __global__
void kernel_deselect_masked(
	GaussianData model,
	uint32_t* selectionMask
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int splatIndex = grid.thread_rank();

	if(splatIndex >= model.count) return;

	uint32_t maskIndex = splatIndex / 32;
	uint32_t bitIndex = splatIndex % 32;
	uint32_t mask = 1 << bitIndex;

	// bool isSet = (selectionMask[maskIndex] & mask ) != 0;

	bool isSet = getBit(selectionMask, splatIndex);

	if(isSet){
		uint32_t flags = model.flags[splatIndex];

		flags = flags & (~FLAGS_SELECTED);

		model.flags[splatIndex] = flags;
	}
}

extern "C" __global__
void kernel_undelete_masked(
	GaussianData model,
	uint32_t* selectionMask
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int splatIndex = grid.thread_rank();

	if(splatIndex >= model.count) return;

	uint32_t maskIndex = splatIndex / 32;
	uint32_t bitIndex = splatIndex % 32;
	uint32_t mask = 1 << bitIndex;

	bool isSet = (selectionMask[maskIndex] & mask ) != 0;

	if(isSet){
		uint32_t flags = model.flags[splatIndex];

		flags = flags & (~FLAGS_DELETED);

		model.flags[splatIndex] = flags;
	}
}

extern "C" __global__
void kernel_brushselect(
	CommonLaunchArgs args, 
	RenderTarget target, 
	GaussianData model,
	uint32_t* selectionMask
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int splatIndex = grid.thread_rank();

	if(splatIndex >= model.count) return;

	auto& mouse = args.mouseEvents;

	uint64_t t_start = nanotime();

	// prepare rays
	vec3 origin = vec3(inverse(target.view) * vec4(0.0, 0.0, 0.0, 1.0));

	vec4 dir_00_projspace = vec4{-1.0f, -1.0f, 1.0f, 1.0f};
	vec4 dir_01_projspace = vec4{-1.0f,  1.0f, 1.0f, 1.0f};
	vec4 dir_10_projspace = vec4{ 1.0f, -1.0f, 1.0f, 1.0f};
	vec4 dir_11_projspace = vec4{ 1.0f,  1.0f, 1.0f, 1.0f};

	float right = 1.0f / target.proj[0][0];
	float up = 1.0f / target.proj[1][1];
	vec4 dir_00_worldspace = inverse(target.view) * vec4(-right, -up, -1.0f, 1.0f);
	vec4 dir_01_worldspace = inverse(target.view) * vec4(-right,  up, -1.0f, 1.0f);
	vec4 dir_10_worldspace = inverse(target.view) * vec4( right, -up, -1.0f, 1.0f);
	vec4 dir_11_worldspace = inverse(target.view) * vec4( right,  up, -1.0f, 1.0f);

	auto getRayDir = [&](float u, float v){
		float A_00 = (1.0f - u) * (1.0f - v);
		float A_01 = (1.0f - u) *         v;
		float A_10 =         u  * (1.0f - v);
		float A_11 =         u  *         v;

		vec3 dir = (
			A_00 * dir_00_worldspace +
			A_01 * dir_01_worldspace +
			A_10 * dir_10_worldspace +
			A_11 * dir_11_worldspace - vec4(origin, 1.0));
		dir = normalize(dir);

		return dir;
	};


	vec3 brushPos = args.state->hovered_pos;
	float distToPos = length(origin - brushPos);

	// handle selection / deselection
	mat4 viewProj = target.proj * target.view;

	uint32_t oldFlags = model.flags[splatIndex];
	uint32_t newFlags = oldFlags;
	vec3 pos = model.position[splatIndex];

	vec3 worldPos = vec3(model.transform * vec4(pos, 1.0f));

	float u = float(mouse.pos_x) / target.width;
	float v = float(mouse.pos_y) / target.height;
	vec3 dir = getRayDir(u, v);

	vec3 camToPoint = worldPos - origin;

	float t = glm::dot(dir, camToPoint);

	vec3 pointOnLine = origin + t * dir;
	float pointToLineDistance = length(pointOnLine - worldPos);

	vec4 ndc = viewProj * vec4(worldPos, 1.0f);
	vec2 imgCoords = vec2(
		(0.5f * (ndc.x / ndc.w) + 0.5f) * target.width,
		(0.5f * (ndc.y / ndc.w) + 0.5f) * target.height
	);


	vec2 mpos = vec2(args.mouseEvents.pos_x, args.mouseEvents.pos_y);
	float d = length(imgCoords - mpos);

	if(t < 0.0f){
		d = Infinity;
	}

	bool splatIntersectsBrush = false;

	if(args.brush.intersectionmode == BRUSH_INTERSECTION_CENTER){
		// INTERSECTION VIA SPLAT POSITION
		splatIntersectsBrush = d < args.brush.size;
	}else if(args.brush.intersectionmode == BRUSH_INTERSECTION_BORDER){
		// INTERSECTION VIA SPLAT BOUNDARY
		if(d < args.brush.size){
			// fast path
			splatIntersectsBrush = true;
		}else{
			mat4 world = model.transform;
			mat4 view = target.view;
			mat4 proj = target.proj;
			mat4 worldView = view * world;
			mat4 worldViewProj = proj * view * world;

			vec4 worldPos = world * vec4(pos, 1.0f);
			vec4 viewPos = view * worldPos;
			vec4 ndc = proj * viewPos;
			vec4 quat = model.quaternion[splatIndex];
			mat3 rotation = glm::mat3_cast(glm::quat(quat.x, quat.y, quat.z, quat.w));
			mat3 scale = mat3(1.0f);
			scale[0][0] = model.scale[splatIndex].x;
			scale[1][1] = model.scale[splatIndex].y;
			scale[2][2] = model.scale[splatIndex].z;

			ndc.x = ndc.x / ndc.w;
			ndc.y = ndc.y / ndc.w;
			ndc.z = ndc.z / ndc.w;

			if(ndc.w < 0.0f) return;
			if(ndc.x < -1.0f || ndc.x >  1.0f) return;
			if(ndc.y < -1.0f || ndc.y >  1.0f) return;

			mat3 cov3D = rotation * scale * scale * transpose(rotation);

			vec2 focal = vec2(
				proj[0][0] * target.width * 0.5f,
				proj[1][1] * target.height * 0.5f
			);
			float s = 1.0f / (viewPos.z * viewPos.z);
			mat3 J = mat3(
				focal.x / viewPos.z   , 0.0f                  , -(focal.x * viewPos.x) * s,
				0.0f                  , focal.y / viewPos.z   , -(focal.y * viewPos.y) * s,
				0.0f                  , 0.0f                  , 0.0f
			);

			mat3 W = transpose(mat3(worldView));
			mat3 T = W * J;

			mat3 cov2Dm = transpose(T) * cov3D * T;
			cov2Dm[0][0] += 0.3f;
			cov2Dm[1][1] += 0.3f;

			vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

			float a = cov2Dv.x;
			float d = cov2Dv.z;
			float b = cov2Dv.y;
			float D = a * d - b * b;
			float trace = a + d;
			float traceOver2 = 0.5f * trace;
			float term2 = sqrt(max(0.05f, traceOver2 * traceOver2 - D));
			float eigenValue1 = traceOver2 + term2;
			float eigenValue2 = traceOver2 - term2;


			// // clip tiny gaussians
			// if(eigenValue1 < 0.05f) return;
			// if(eigenValue2 < 0.05f) return;

			// if(args.uniforms.makePoints){
			// 	if(eigenValue1 < 0.105f) return;
			// 	if(eigenValue2 < 0.105f) return;
			// }

			vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
			vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

			float splatScale = args.uniforms.splatSize;
			const float sqrt8 = sqrt(8.0f);
			float pixels_a = sqrt8 * sqrt(eigenValue1);
			float pixels_b = sqrt8 * sqrt(eigenValue2);
			vec2 basisVector1 = eigenVector1 * splatScale * min(pixels_a, MAX_SCREENSPACE_SPLATSIZE);
			vec2 basisVector2 = eigenVector2 * splatScale * min(pixels_b, MAX_SCREENSPACE_SPLATSIZE);

			{ 
				float diag = args.brush.size * 2.0f;
				float a = length(basisVector1);
				float b = length(basisVector2);
				float a_outer = (a + diag / 2);
				float b_outer = (b + diag / 2);

				splatIntersectsBrush = intersection_point_splat(mpos, imgCoords, 
					a_outer * basisVector1 / a, 
					b_outer * basisVector2 / b
				);
			}

			if(pixels_a < args.brush.minSplatSize && pixels_b < args.brush.minSplatSize){
				splatIntersectsBrush = false;
			}
		
		}
	}

	if(splatIntersectsBrush){
		newFlags = newFlags | FLAGS_HIGHLIGHTED;
	}else{
		newFlags = newFlags & (~FLAGS_HIGHLIGHTED);
	}

	if(args.brush.mode == BRUSHMODE::SELECT){
		bool isSelected = splatIntersectsBrush && args.mouseEvents.isLeftDown;

		if(isSelected){
			newFlags = newFlags | FLAGS_SELECTED;
		}
	}else if(args.brush.mode == BRUSHMODE::REMOVE_FLAGS){
		bool isSelected = splatIntersectsBrush && args.mouseEvents.isLeftDown;

		if(isSelected){
			newFlags = newFlags & (~0b1001);
		}
	}else if(args.brush.mode == BRUSHMODE::ERASE){
		bool isSelected = splatIntersectsBrush && args.mouseEvents.isLeftDown;
		bool wasSelected = (newFlags & FLAGS_DELETED);

		if(!wasSelected && isSelected){
			newFlags = newFlags | FLAGS_DELETED;
			uint32_t maskIndex = splatIndex / 32;
			uint32_t bitIndex = splatIndex % 32;
			uint32_t mask = 1 << bitIndex;

			if(selectionMask){
				atomicOr(&selectionMask[maskIndex], mask);
			}
		}
	}else if(args.brush.mode == BRUSHMODE::MULTIPLY){

		// if(splatIntersectsBrush){
		// 	vec4 color = model.color_painting[splatIndex];
		// 	// uint16_t* rgba = (uint16_t*)&color;
		// 	color.r = color.r * 0.98f;
		// 	color.g = color.g * 0.98f;
		// 	color.b = color.b * 0.98f;

		// 	model.color_painting[splatIndex] = color;
		// }
	}else if(args.brush.mode == BRUSHMODE::ADD){

		// if(splatIntersectsBrush){
		// 	vec4 color = model.color_painting[splatIndex];
		// 	// uint16_t* rgba = (uint16_t*)&color;
		// 	// rgba[0] = clamp(rgba[0] + 1, 0, 65535);
		// 	// rgba[1] = clamp(rgba[1] + 1, 0, 65535);
		// 	// rgba[2] = clamp(rgba[2] + 1, 0, 65535);
		// 	color.r = clamp(color.r + 0.01f, 0.0f, 1.0f);
		// 	color.g = clamp(color.g + 0.01f, 0.0f, 1.0f);
		// 	color.b = clamp(color.b + 0.01f, 0.0f, 1.0f);

		// 	model.color_painting[splatIndex] = color;
		// }
	}else{
		// flag = flag & (~0b110);
		newFlags = newFlags & ~FLAGS_HIGHLIGHTED;
		newFlags = newFlags & ~FLAGS_HIGHLIGHTED_NEGATIVE;
	}

	model.flags[splatIndex] = newFlags;

	bool wasSelected = oldFlags & FLAGS_SELECTED;
	bool isSelected = newFlags & FLAGS_SELECTED;

	if(isSelected && !wasSelected){
		if(selectionMask){
			setBit(selectionMask, splatIndex);
		}
		atomicAdd(&args.state->numSelectedSplats, 1);
	}else if(!isSelected && wasSelected){
		// uint32_t maskIndex = splatIndex / 32;
		// uint32_t bitIndex = splatIndex % 32;
		// uint32_t mask = 1 << bitIndex;

		if(selectionMask){
			// atomicAnd(&selectionMask[maskIndex], ~mask);
			clearBit(selectionMask, splatIndex);
		}
		atomicSub(&args.state->numSelectedSplats, 1);
	}
}



extern "C" __global__
void kernel_compute_boundingbox2(CommonLaunchArgs args, GaussianData model, Box3& box, bool onlySelected){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	
	int numSplatsPerThread = (model.count + grid.num_threads() - 1) / grid.num_threads();
	int numSplatsPerBlock = numSplatsPerThread * block.num_threads();

	int index = grid.thread_rank();

	vec3 w_min = {Infinity, Infinity, Infinity};
	vec3 w_max = {-Infinity, -Infinity, -Infinity};

	for(int i = 0; i < numSplatsPerThread; i++){

		int firstSplatOfBlock = grid.block_rank() * numSplatsPerBlock;

		int splatIndex = firstSplatOfBlock + i * block.num_threads() + block.thread_rank();

		if(splatIndex >= model.count) continue;

		bool deleted = model.flags[splatIndex] & FLAGS_DELETED;
		bool selected = model.flags[splatIndex] & FLAGS_SELECTED;

		if(deleted) continue;
		if(onlySelected && !selected) continue;

		vec3 pos = vec3(model.transform * vec4(model.position[splatIndex], 1.0f));

		w_min.x = min(w_min.x, pos.x);
		w_min.y = min(w_min.y, pos.y);
		w_min.z = min(w_min.z, pos.z);
		w_max.x = max(w_max.x, pos.x);
		w_max.y = max(w_max.y, pos.y);
		w_max.z = max(w_max.z, pos.z);
	}

	atomicMinFloat(&box.min.x, w_min.x);
	atomicMinFloat(&box.min.y, w_min.y);
	atomicMinFloat(&box.min.z, w_min.z);
	atomicMaxFloat(&box.max.x, w_max.x);
	atomicMaxFloat(&box.max.y, w_max.y);
	atomicMaxFloat(&box.max.z, w_max.z);
}

extern "C" __global__
void kernel_compute_boundingbox(CommonLaunchArgs args, GaussianData model, vec3& min, vec3& max){

	auto block = cg::this_thread_block();

	int index = cg::this_grid().thread_rank();

	vec3 pos = {0, 0, 0};

	bool deleted = model.flags[index] & FLAGS_DELETED;

	if(index < model.count && !deleted){
		pos = vec3(model.transform * vec4(model.position[index], 1.0f));
	};

	int lane = threadIdx.x % warpSize;

	float warpMinX = warpReduceMin(pos.x);
	float warpMinY = warpReduceMin(pos.y);
	float warpMinZ = warpReduceMin(pos.z);

	float warpMaxX = warpReduceMax(pos.x);
	float warpMaxY = warpReduceMax(pos.y);
	float warpMaxZ = warpReduceMax(pos.z);

	if(index == 0){
		min.x = 0.0f;
		min.y = 0.0f;
		min.z = 0.0f;
	}

	float blockMaxX = blockReduceMax(pos.x);
	float blockMaxY = blockReduceMax(pos.y);
	float blockMaxZ = blockReduceMax(pos.z);

	if(block.thread_rank() == 0){
		atomicMaxFloat(&max.x, blockMaxX);
		atomicMaxFloat(&max.y, blockMaxY);
		atomicMaxFloat(&max.z, blockMaxZ);
	}
}



// Bubble sort!
// More specififically odd-even sort, see https://en.wikipedia.org/wiki/Odd%E2%80%93even_sort
// But it's basically parallel bubble-sort, and it's nice being able to claim you're using bubble sort for real.
void bubbleSort(uint64_t* list, int count){

	auto block = cg::this_thread_block();

	auto swap = [&](int i, int j){
		uint64_t tmp = list[i];
		list[i] = list[j];
		list[j] = tmp;
	};

	__shared__ bool sh_sorted;
	sh_sorted = false;

	block.sync();

	int safeguard = 0;
	while(true){

		sh_sorted = true;

		block.sync();

		// sort odd-even pairs [1, 2] [3, 4], ...
		for(
			int i = 2 * block.thread_rank() + 1;
			i < count - 1;
			i += block.num_threads()
		){

			if(list[i] > list[i + 1]){
				swap(i, i + 1);
				sh_sorted = false;
			}
		}

		block.sync();

		// sort even-odd pairs [0, 1], [2, 3] ...
		for(
			int i = 2 * block.thread_rank() + 0;
			i < count - 1;
			i += block.num_threads()
		){

			if(list[i] > list[i + 1]){
				swap(i, i + 1);
				sh_sorted = false;
			}
		}

		block.sync();

		if(sh_sorted) break;

		safeguard++;
		if(safeguard > 1000){
			printf("exceeded safeguard: %u, ... count: %u \n", safeguard, count);

			return;
		}
	}

}

extern "C" __global__
void kernel_computeHoveredObject(CommonLaunchArgs args, RenderTarget target){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// if(grid.thread_rank() > 0) return;

	if(grid.thread_rank() == 0)
	{
		int px = clamp(args.mouseEvents.pos_x, 0.0f, target.width - 1.0f);
		int py = clamp(args.mouseEvents.pos_y, 0.0f, target.height - 1.0f);

		int pixelID = px + target.width * py;

		uint64_t pixel = target.framebuffer[pixelID];
		uint32_t udepth = pixel >> 32;
		float depth = __int_as_float(udepth);
		// vec3 origin = vec3(inverse(target.view) * vec4(0.0, 0.0, 0.0, 1.0));

		uint64_t C = pixel & 0xffffffff;
		uint8_t* rgba = (uint8_t*)&C;
		vec4 color = {rgba[0], rgba[1], rgba[2], rgba[3]};

		args.state->hovered_depth = depth;
		args.state->hovered_color = color;

		// hovered_pos from pixel coords and screen-space depth.
		// (screen-space depth is "flat" and does not represent euclidean distance to fragment)
		vec4 i_ndc = {
			(2.0f * float(px) / target.width - 1.0f) * depth,
			(2.0f * float(py) / target.height - 1.0f) * depth,
			0.0f,
			depth
		};
		vec4 i_viewpos = glm::inverse(target.proj) * i_ndc;
		i_viewpos.w = 1.0f;
		vec4 i_worldpos = glm::inverse(target.view) * i_viewpos;
		args.state->hovered_pos = vec3(i_worldpos);
	}


	__shared__ uint64_t sh_depths[512];
	__shared__ uint32_t sh_count;
	
	if(grid.block_rank() == 0){

		sh_count = 0;

		int px = args.mouseEvents.pos_x;
		int py = args.mouseEvents.pos_y;
		int windowRadius = 8;
		int windowWidth = 2 * windowRadius + 1;
		int windowPixels = windowWidth * windowWidth;
		ivec2 window_min = ivec2{px, py} - windowRadius;
		ivec2 window_max = ivec2{px, py} + windowRadius;

		int pixels_x = window_max.x - window_min.x + 1;
		int pixels_y = window_max.y - window_min.y + 1;

		block.sync();

		// load pixel/depth values into shared memory
		for(
			int i = block.thread_rank();
			i < windowPixels;
			i += block.num_threads()
		){

			int ox = i % windowWidth - windowRadius;
			int oy = i / windowWidth - windowRadius;

			int pixel_x = px + ox;
			int pixel_y = py + oy;
			int pixelID = pixel_x + target.width * pixel_y;

			if(pixel_x < 0 || pixel_x >= target.width) continue;
			if(pixel_y < 0 || pixel_y >= target.height) continue;

			uint64_t pixel = target.framebuffer[pixelID];
			uint32_t udepth = pixel >> 32;
			float depth = __int_as_float(udepth);

			if(depth != Infinity && depth > 0.0f){
				uint32_t index = atomicAdd(&sh_count, 1);
				sh_depths[index] = pixel;
			}
		}

		block.sync();

		bubbleSort(&sh_depths[0], sh_count);

		block.sync();

		// take median
		uint64_t pixel = sh_depths[sh_count / 2];
		uint32_t udepth = pixel >> 32;
		float depth = __int_as_float(udepth);

		args.state->hovered_depth = depth;

		// Our projection matrix is slightly unconventionall, but also easy to reverse because of that. 
		// see GLRenderer.h - Camera::createProjectionMatrix()
		// The projection is basically a dot product of viewPos and vec4(f/aspect, f, 0.0f, -1.0f),
		// followed by the usual division of ndc.xy / ndc.w. In our case, ndc.w is the linear depth,
		// so we can multiply ndc.xy by depth, and also enter -depth for the linear depth in view space along -z. 
		vec4 i_viewpos = {
			((2.0f * float(px) / target.width - 1.0f) * depth) / target.proj[0][0],
			((2.0f * float(py) / target.height - 1.0f) * depth) / target.proj[1][1],
			-depth,
			1.0f
		};
		vec4 i_worldpos = glm::inverse(target.view) * i_viewpos;
		args.state->hovered_pos = vec3(i_worldpos);

		if(__float_as_uint(depth) == Infinity){
			args.state->hovered_pos = {0.0f, 0.0f, 0.0f};
		}
	}


}


extern "C" __global__
void kernel_compute_mortoncodes_2x32bit(GaussianData splats, Box3 aabb, uint32_t* mortoncodes_lower, uint32_t* mortoncodes_higher, uint32_t* ordering){

	auto grid = cg::this_grid();
	uint32_t index = grid.thread_rank();

	if(index >= splats.count) return;
	

	// from: https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
	// license: Creative Commons Attribute-NonCommercial Sharealike 3.0 Unported license.
	// also: https://github.com/Forceflow/libmorton

	// method to seperate bits from a given integer 3 positions apart
	auto splitBy3 = [](uint32_t a) -> uint64_t {
		uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
		
		x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
		x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
		x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
		x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
		x = (x | x << 2) & 0x1249249249249249;
		
		return x;
	};

	auto encode = [&splitBy3](uint32_t x, uint32_t y, uint32_t z) -> uint64_t {
		uint64_t answer = 0;
		answer |= (splitBy3(x) << 2) | (splitBy3(y) << 1) | (splitBy3(z) << 0);
		
		return answer;
	};

	vec3 size = aabb.max - aabb.min;
	float maxsize = max(max(size.x, size.y), size.z);

	float factor = 1'048'576.0f; // 2^20

	vec3 position = splats.position[index];
	uvec3 upos = ((position - aabb.min) / maxsize) * factor;

	// if(index == 0){
	// 	printf("splats.count: %d \n", splats.count);
	// 	printf("%.1f, %.1f, %.1f \n", aabb.min.x, aabb.min.y, aabb.min.z);
	// }

	uint64_t mortoncode = encode(upos.x, upos.y, upos.z);

	mortoncodes_lower[index] = mortoncode & 0xffffffff;
	mortoncodes_higher[index] = (mortoncode >> 32) & 0xffffffff;

	// mortoncodes[index] = mortoncode;
	ordering[index] = index;
}

extern "C" __global__
void kernel_compute_mortoncodes_32bit(GaussianData splats, Box3 aabb, uint32_t* mortoncodes, uint32_t* ordering){

	auto grid = cg::this_grid();
	uint32_t index = grid.thread_rank();

	if(index >= splats.count) return;
	

	// from: https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
	// license: Creative Commons Attribute-NonCommercial Sharealike 3.0 Unported license.
	// also: https://github.com/Forceflow/libmorton

	// method to seperate bits from a given integer 3 positions apart
	auto splitBy3 = [](uint32_t a) -> uint64_t {
		uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
		
		x = (x | x << 32) & 0x001f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
		x = (x | x << 16) & 0x001f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
		x = (x | x <<  8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
		x = (x | x <<  4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
		x = (x | x <<  2) & 0x1249249249249249;
		
		return x;
	};

	auto encode = [&splitBy3](uint32_t x, uint32_t y, uint32_t z) -> uint64_t {
		uint64_t answer = 0;
		answer |= (splitBy3(x) << 2) | (splitBy3(y) << 1) | (splitBy3(z) << 0);
		
		return answer;
	};

	vec3 size = aabb.max - aabb.min;
	float maxsize = max(max(size.x, size.y), size.z);

	float factor = 1024.0f;

	vec3 position = splats.transform * vec4(splats.position[index], 1.0f);
	uvec3 upos = ((position - aabb.min) / maxsize) * factor;

	// if(index == 0){
	// 	printf("splats.count: %d \n", splats.count);
	// 	printf("%.1f, %.1f, %.1f \n", aabb.min.x, aabb.min.y, aabb.min.z);
	// }

	uint64_t mortoncode = encode(upos.x, upos.y, upos.z);

	// mortoncode = (mortoncode >> 18) << 18;
	// mortoncode = mortoncode << 18;

	// if(length(splats.scale[index]) > 30.0f){
	// 	// // mortoncode = mortoncode | (1 << 62);
	// 	// mortoncode = 0;
	// }

	// mortoncodes_lower[index] = mortoncode & 0xffffffff;
	// mortoncodes_higher[index] = (mortoncode >> 32) & 0xffffffff;

	mortoncodes[index] = mortoncode;
	ordering[index] = index;
}

// set deletionlist[splatIndex] to 0xffffffff if splat is flagged for removal
extern "C" __global__
void kernel_compute_deletionlist(GaussianData splats, uint32_t* deletionlist){

	auto grid = cg::this_grid();
	uint32_t index = grid.thread_rank();

	if(index >= splats.count) return;

	if(splats.flags[index] & FLAGS_DELETED){
		deletionlist[index] = 0xffffffff;
	}
}

extern "C" __global__
void kernel_downsample(GaussianData source, GaussianData target, void* ptr_sampleGrid, uint32_t* counter){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint32_t sourceIndex = grid.thread_rank();

	int sampleGridSize = 256;
	float fSampleGridSize = sampleGridSize;

	vec3 aabbSize = source.max - source.min;
	vec3 gridMin = source.min;
	float gridSize = max(max(aabbSize.x, aabbSize.y), aabbSize.z);

	struct RGBA{
		float r, g, b, a;
	};

	auto toCellID = [&](int X, int Y, int Z){
		uint32_t cellID = X + Y * sampleGridSize + Z * sampleGridSize * sampleGridSize;
		cellID = clamp(cellID, 0u, sampleGridSize * sampleGridSize * sampleGridSize - 1u);

		return cellID;
	};

	auto toCellCoord = [&](int cellID){
		int Z = cellID / (sampleGridSize * sampleGridSize);
		int Y = (cellID % (sampleGridSize * sampleGridSize)) / sampleGridSize;
		int X = cellID % sampleGridSize;

		return ivec3(X, Y, Z);
	};

	RGBA* sampleGrid = (RGBA*)ptr_sampleGrid;
	float* sampleGridf = (float*)ptr_sampleGrid;
	uint32_t* sampleGrid_u = (uint32_t*)ptr_sampleGrid;

	// // CLEAR SAMPLE GRID
	// process(4 * sampleGridSize * sampleGridSize * sampleGridSize, [&](int index){
	// 	sampleGridf[index] = 0.0f;
	// });
	
	// grid.sync();

	// // sample splats
	// process(source.count, [&](int index){

	// 	vec3 pos = source.transform * vec4(source.position[index], 1.0f);
	// 	vec4 color = source.color[index];
	// 	// uint16_t* rgba = (uint16_t*)&color;
	// 	vec3 scale = source.scale[index];

	// 	vec3 samplePos = fSampleGridSize * (pos - source.min) / gridSize;
	// 	ivec3 sampleCoord = (ivec3)samplePos;

	// 	uint32_t cellIndex = toCellID(sampleCoord.x, sampleCoord.y, sampleCoord.z);

	// 	float w_scale = clamp(scale.x * scale.y * scale.z * 100'000.0f, 0.001f, 100.0f);
	// 	uint32_t weight = 255.0f * float(rgba[3]) * w_scale;

	// 	atomicAdd(&sampleGrid_u[4 * cellIndex + 0], weight * uint32_t(rgba[0]));
	// 	atomicAdd(&sampleGrid_u[4 * cellIndex + 1], weight * uint32_t(rgba[1]));
	// 	atomicAdd(&sampleGrid_u[4 * cellIndex + 2], weight * uint32_t(rgba[2]));
	// 	atomicAdd(&sampleGrid_u[4 * cellIndex + 3], weight);
	// });

	// grid.sync();

	// // EXTRACT SPLATS FROM SAMPLE GRID
	// process(sampleGridSize * sampleGridSize * sampleGridSize, [&](int cellIndex){
	// 	uint32_t R = sampleGrid_u[4 * cellIndex + 0];
	// 	uint32_t G = sampleGrid_u[4 * cellIndex + 1];
	// 	uint32_t B = sampleGrid_u[4 * cellIndex + 2];
	// 	uint32_t A = sampleGrid_u[4 * cellIndex + 3];

	// 	if(A > 100){
	// 		uint32_t targetIndex = atomicAdd(counter, 1);

	// 		if(targetIndex < 1'000'000){

	// 			uint64_t color;
	// 			uint16_t* rgba = (uint16_t*)&color;
	// 			rgba[0] = R / A;
	// 			rgba[1] = G / A;
	// 			rgba[2] = B / A;
	// 			rgba[3] = 65535;

	// 			ivec3 cellCoord = toCellCoord(cellIndex);
	// 			vec3 pos = gridSize * vec3(cellCoord) / fSampleGridSize + source.min;

	// 			target.position[targetIndex]   = pos;
	// 			target.scale[targetIndex]      = vec3{1.0f, 1.0f, 1.0f} * 0.005f * 2.0f;
	// 			target.quaternion[targetIndex] = {1.0f, 0.0f, 0.0f, 0.0f};
	// 			target.color[targetIndex]      = color;
	// 		}
	// 	}
	// });



}

// Filters splats in source and writes those that pass in target.
// - If target.position is nullptr, it just counts without storing splats
// - Splats are stored at splats[atomicAdd(counter, 1)].
//   This means that if the target already has n splats, then the counter must be initialized to n.
extern "C" __global__
void kernel_filter(
	GaussianData source, 
	GaussianData target, 
	FilterRules rules, 
	uint32_t* counter
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	process(source.count, [&](int sourceIndex){

		vec3 pos            = source.position[sourceIndex];
		vec3 scale          = source.scale[sourceIndex];
		vec4 quaternion     = source.quaternion[sourceIndex];
		Color color         = source.color[sourceIndex];
		uint32_t flags      = source.flags[sourceIndex];

		bool isSelected = flags & FLAGS_SELECTED;
		bool isDeleted = flags & FLAGS_DELETED;

		bool acceptSplat = true;

		if(rules.selection == FILTER_SELECTION_SELECTED){
			acceptSplat = acceptSplat && isSelected;
		}else if(rules.selection == FILTER_SELECTION_UNSELECTED){
			acceptSplat = acceptSplat && !isSelected;
		}

		if(rules.deleted == FILTER_DELETED_NONDELETED){
			acceptSplat = acceptSplat && !isDeleted;
		}else if(rules.deleted == FILTER_DELETED_DELETED){
			acceptSplat = acceptSplat && isDeleted;
		}

		if(acceptSplat){
			uint32_t targetIndex = atomicAdd(counter, 1);

			// if position is null, we just count but don't copy splats.
			if(target.position){
				target.position[targetIndex]   = pos;
				target.scale[targetIndex]      = scale;
				target.quaternion[targetIndex] = quaternion;
				target.color[targetIndex]      = color;
				target.flags[targetIndex]      = flags;
			}
		}

	});
}

extern "C" __global__
void kernel_setFlags(GaussianData model, uint32_t newFlags){
	process(model.count, [&](int index){
		model.flags[index] = newFlags;
	});
}

extern "C" __global__
void kernel_memset_u64(uint64_t* data, uint64_t value, uint64_t count){
	uint64_t index = cg::this_grid().thread_rank();

	if(index >= count) return;
	
	data[index] = value;
}