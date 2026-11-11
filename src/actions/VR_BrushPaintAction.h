
#pragma once

#include "InputAction.h"

struct VR_BrushPaintUndoAction : public Action{
	
	struct UndoData{
		uint32_t splatcount;
		uint32_t diffcount;
		CUdeviceptr cptr_stashedColors;  // copy of colors before modification. Removed after compaction.
		CUdeviceptr cptr_indices;        // indices of modified splats. Generated during compaction.
		CUdeviceptr cptr_indexedColors;  // colors of modified splats. Generated during compaction.
	};

	unordered_map<SNSplats*, UndoData> undodatas;

	~VR_BrushPaintUndoAction(){
		for(auto [node, undodata] : undodatas){
			if(undodata.cptr_indices)       CURuntime::free(undodata.cptr_indices);
			if(undodata.cptr_indexedColors) CURuntime::free(undodata.cptr_indexedColors);
		}
	}

	void undo(){
		auto editor = SplatEditor::instance;

		for(auto [node, undodata] : undodatas){

			GaussianData data = node->dmng.data;

			int64_t colorBufferByteSize = sizeof(Color) * undodata.splatcount;

			// to undo, we swap colors from the model's buffer with the indexed stashed colors.
			void* args[] = {&undodata.cptr_indices, &undodata.cptr_indexedColors, &data.color, &undodata.diffcount};
			editor->prog_gaussians_editing->launch("kernel_color_diff_swap", args, undodata.diffcount);
		}
	}

	void redo(){
		auto editor = SplatEditor::instance;

		for(auto [node, undodata] : undodatas){
			GaussianData data = node->dmng.data;

			// to redo, we again swap colors from the model's buffer with the indexed stashed colors.
			void* args[] = {&undodata.cptr_indices, &undodata.cptr_indexedColors, &data.color, &undodata.diffcount};
			editor->prog_gaussians_editing->launch("kernel_color_diff_swap", args, undodata.diffcount);
		}
	}

	// turn a copy of the color buffer into a diff of modified splats.
	void compaction(){
		auto editor = SplatEditor::instance;

		for(auto& [node, undodata] : undodatas){

			GaussianData data = node->dmng.data;

			CUdeviceptr cptr_counter = CURuntime::alloc("counter", 4);
			cuMemsetD32(cptr_counter, 0, 1);

			void* argsCount[] = {&undodata.cptr_stashedColors, &data.color, &data.count, &cptr_counter};
			editor->prog_gaussians_editing->launch("kernel_countChangedColors", argsCount, undodata.splatcount);

			cuMemcpyDtoH(&undodata.diffcount, cptr_counter, 4);

			if (undodata.diffcount > 0) {

				undodata.cptr_indices = CURuntime::alloc("color diff indices", 4 * undodata.diffcount);
				undodata.cptr_indexedColors = CURuntime::alloc("color diff colors", sizeof(Color) * undodata.diffcount);

				cuMemsetD32(cptr_counter, 0, 1);
			
				void* args[] = {
					&undodata.cptr_stashedColors, &data.color, &data.count,
					&undodata.cptr_indices, &undodata.cptr_indexedColors, &cptr_counter
				};

				editor->prog_gaussians_editing->launch("kernel_create_color_diff", args, undodata.splatcount);
			}else{
				undodata.cptr_indices = 0;
				undodata.cptr_indexedColors = 0;
			}

			CURuntime::free(cptr_counter);
			
			// no longer needed after compaction
			CURuntime::free(undodata.cptr_stashedColors); 
			undodata.cptr_stashedColors = 0;
		}
	}

	int64_t byteSize(){
		
		int64_t totalSize = 0;
		for(auto [node, undodata] : undodatas){
			if(undodata.cptr_stashedColors)  totalSize += CURuntime::getSize(undodata.cptr_stashedColors);
			if(undodata.cptr_indices)        totalSize += CURuntime::getSize(undodata.cptr_indices);
			if(undodata.cptr_indexedColors)  totalSize += CURuntime::getSize(undodata.cptr_indexedColors);
		}

		return totalSize;
	}
};

struct VR_BrushPaintAction : public InputAction{

	shared_ptr<VR_BrushPaintUndoAction> currentUndoAction = nullptr;

	const dmat4 flip = glm::dmat4(
		1.0,  0.0, 0.0, 0.0,
		0.0,  0.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 1.0
	);

	void start(){
		println("starting VR_BrushPaintAction");

	}

	void paintBrushStep(vec3 brushPos, float brushRadius){
		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		int numTicks = 1;

		editor->scene.process<SNSplats>([&](SNSplats* node) {
			if(!node->visible) return;
			if(node->locked) return;

			auto data = node->dmng.data;
			data.transform = node->transform_global;

			auto undodata = currentUndoAction->undodatas[node];
			CUdeviceptr cptr_stashedColors = undodata.cptr_stashedColors;

			
			void* args[] = {&editor->launchArgs, &data, &cptr_stashedColors, &brushPos, &brushRadius, &settings.brush.color, &numTicks};
			editor->prog_gaussians_editing->launch("kernel_paint_sphere", args, data.count);
		});
		
	}

	void update(){
		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;
		Scene& scene = editor->scene;
		auto ovr = editor->ovr;

		if(!ovr->isActive()) return;

		Controller right = ovr->getRightController();

		const float MIN_STEPDISTANCE = 3.0f;
		// do a brush step every <stepdistance> meters
		float brushRadius = settings.vr_brushSize;
		float stepdistance = brushRadius * 0.1f;

		static vec3 lastBrushPos = {0.0f, 0.0f, 0.0f};
		static float accumulatedDistance = 0.0f;

		mat4 mTranslate = translate(vec3{0.0f, settings.vr_brushSize, 0.0f});
		mat4 mScale = scale(vec3{1.0f, 1.0f, 1.0f} * settings.vr_brushSize);
		mat4 mRot = glm::rotate(-140.0f * 3.1415f / 180.0f, vec3{ 1.0f, 0.0f, 0.0f });
		mat4 transform = mat4(ovr->flip * right.transform) * mRot * mTranslate * mScale;
		vec3 brushPos = transform * vec4{0.0f, 0.0f, 0.0f, 1.0f};
		vec4 color = settings.brush.color;

		if(right.triggered()){
			beginUndoable();
			lastBrushPos = brushPos;
			accumulatedDistance = 0.0f;
		}else if (right.untriggered()) {
			endUndoable();
		}

		if(right.isTriggerPressed()){
			// Using same logic as mouse brushes, but in 3D
			//
			//
			//   "unrolled" origin                                              
			//    |                                                       current mouse pos
			//    |            previous mouse pos                                  |
			//    |              |                                                 |
			//    O--------------a-------------------------------------------------b
			//    |- acc. dist. -|
			//    |              |-------------------   50 pixels  ----------------|
			//    S                   S                   S                   S                   S
			//    |---- 20 pixels ----|---- 20 pixels ----|---- 20 pixels ----|---- 20 pixels ----|
			// 
			// stepsize: 20 pixels
			// O: The unrolled origin is given by the accumulated mouse movement since the last brush step, 
			//        straightened onto the line between previous frame's mouse pos and current frame's mouse pos.
			// S: A brush "step" at which we paint. If the mouse moves 50 pixel and we draw every 20 pixels, 
			//        then there will be 2 or 3 brush steps in a single frame.
			// 

			vec3 a = lastBrushPos;
			vec3 b = brushPos;
			vec3 ab = b - a;
			vec3 dir_ab = normalize(ab);
			vec3 O = a - accumulatedDistance * dir_ab;
			// float distance = length(b - a);
			float dist_Ob = length(b - O);

			if(right.triggered()){
				vec3 brushPos = b;
				paintBrushStep(brushPos, brushRadius);
			}else if(length(b - a) > 0.001f){

				// println("================================");
				// println("a: {:6.1f}, {:6.1f}", a.x, a.y);
				// println("b: {:6.1f}, {:6.1f}", b.x, b.y);
				// println("O: {:6.1f}, {:6.1f}", O.x, O.y);
				// println("accumulatedDistance: {:.1f}", accumulatedDistance);
				// println("dist_Ob:             {:.1f}", dist_Ob);
				// println("stepdistance:        {:.1f}", stepdistance);

				vec3 lastBrushedPos = O;
				for(float t = stepdistance; t < dist_Ob; t += stepdistance){
					vec3 brushPos = O + t * dir_ab;

					paintBrushStep(brushPos, brushRadius);

					lastBrushedPos = brushPos;
				}
				
				lastBrushPos = b;
				accumulatedDistance = length(b - lastBrushedPos);
			}
		}

		{ // Place Brush Cage
			mat4 mTranslate = translate(vec3{0.0f, settings.vr_brushSize, 0.0f});
			mat4 mScale = scale(vec3{1.0f, 1.0f, 1.0f} * settings.vr_brushSize);
			mat4 mRot = glm::rotate(-140.0f * 3.1415f / 180.0f, vec3{ 1.0f, 0.0f, 0.0f });
			mat4 transform_controller = mat4(flip * right.transform) * mRot * mTranslate * mScale;

			editor->sn_brushsphere->transform = transform_controller;
			editor->sn_brushsphere->locked = true;
			editor->sn_brushsphere->visible = true;
			
			// set brush color (specifically made for 4*16 bit colors)
			static_assert(sizeof(Color) == 8);
			Color value = Color::fromNormalized(settings.brush.color);
			Color* destination = editor->sn_brushsphere->dmng.data.color;
			uint32_t count = editor->sn_brushsphere->dmng.data.count;
			
			void* args[] = {&destination, &value, &count};
			editor->prog_gaussians_editing->launch("kernel_memset_u64", args, count);
		}

	}

	void stop(){
		auto editor = SplatEditor::instance;

		println("stopping VR_BrushSelectAction");
		endUndoable();
		editor->sn_brushsphere->visible = false;
	}

	void beginUndoable(){
		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		if(currentUndoAction){
			currentUndoAction->compaction();
			currentUndoAction = nullptr;
		}

		currentUndoAction = make_shared<VR_BrushPaintUndoAction>();
		
		// create undo/redo buffers
		editor->scene.process<SNSplats>([&](SNSplats* node) {

			if(!node->visible) return;
			if(node->locked) return;

			auto data = node->dmng.data;
			data.transform = node->transform_global;

			uint32_t requiredBits = node->dmng.data.count;
			uint32_t requiredBytes = (requiredBits / 8) + 4;

			int64_t colorBufferByteSize = sizeof(Color) * data.count;

			CUdeviceptr cptr_stashedColors = CURuntime::alloc("PaintAction original colors", colorBufferByteSize);
			cuMemcpy(cptr_stashedColors, (CUdeviceptr)data.color, colorBufferByteSize);

			VR_BrushPaintUndoAction::UndoData undodata;
			undodata.splatcount = data.count;
			undodata.cptr_stashedColors = cptr_stashedColors;
			currentUndoAction->undodatas[node] = undodata;
		});
	}

	void endUndoable(){
		auto editor = SplatEditor::instance;

		if(currentUndoAction){

			currentUndoAction->compaction();

			editor->addAction(currentUndoAction);

			currentUndoAction = nullptr;
		}
	}

};
