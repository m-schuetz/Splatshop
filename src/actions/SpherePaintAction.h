#pragma once

#include "InputAction.h"

struct PaintUndoAction : public Action{

	struct UndoData{
		uint32_t splatcount;
		uint32_t diffcount;
		CUdeviceptr cptr_stashedColors;  // copy of colors before modification. Removed after compaction.
		CUdeviceptr cptr_indices;        // indices of modified splats. Generated during compaction.
		CUdeviceptr cptr_indexedColors;  // colors of modified splats. Generated during compaction.
	};

	unordered_map<SNSplats*, UndoData> undodatas;

	~PaintUndoAction(){
		for(auto [node, undodata] : undodatas){
			if(undodata.cptr_indices)       CURuntime::free(undodata.cptr_indices);
			if(undodata.cptr_indexedColors) CURuntime::free(undodata.cptr_indexedColors);
		}
	}

	void undo(){
		auto editor = SplatEditor::instance;

		float t_start = 0.0f;
		static CUevent ce_start = 0;
		static CUevent ce_end = 0;

		if(ce_start == 0){
			cuEventCreate(&ce_start, CU_EVENT_DEFAULT);
			cuEventCreate(&ce_end, CU_EVENT_DEFAULT);
		}

		if(Runtime::measureTimings){
			cuEventRecord(ce_start, 0);
			t_start = now();
		}

		for(auto [node, undodata] : undodatas){

			GaussianData data = node->dmng.data;

			int64_t colorBufferByteSize = sizeof(Color) * undodata.splatcount;

			// to undo, we swap colors from the model's buffer with the indexed stashed colors.
			void* args[] = {&undodata.cptr_indices, &undodata.cptr_indexedColors, &data.color, &undodata.diffcount};
			editor->prog_gaussians_editing->launch("kernel_color_diff_swap", args, undodata.diffcount);
		}

		if(Runtime::measureTimings){
			cuEventRecord(ce_end, 0);
			float duration_host = now() - t_start;

			cuCtxSynchronize();

			float duration_device = 0.0f;
			cuEventElapsedTime(&duration_device, ce_start, ce_end);

			println("PaintUndoAction::undo() timings: host: {:.3f} ms, device: {:.3f} ms", duration_host, duration_device);
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

		float t_start = 0.0f;
		static CUevent ce_start = 0;
		static CUevent ce_end = 0;

		if(ce_start == 0){
			cuEventCreate(&ce_start, CU_EVENT_DEFAULT);
			cuEventCreate(&ce_end, CU_EVENT_DEFAULT);
		}

		if(Runtime::measureTimings){
			cuEventRecord(ce_start, 0);
			t_start = now();
		}

		CUdeviceptr cptr_counter = CURuntime::alloc("counter", 4);

		for(auto& [node, undodata] : undodatas){

			GaussianData data = node->dmng.data;

			cuMemsetD32(cptr_counter, 0, 1);

			void* argsCount[] = {&undodata.cptr_stashedColors, &data.color, &data.count, &cptr_counter};
			editor->prog_gaussians_editing->launch("kernel_countChangedColors", argsCount, undodata.splatcount);

			cuMemcpyDtoH(&undodata.diffcount, cptr_counter, 4);

			if(undodata.diffcount > 0){
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

			
			// no longer needed after compaction
			CURuntime::free(undodata.cptr_stashedColors); 
			undodata.cptr_stashedColors = 0;
		}

		CURuntime::free(cptr_counter);

		if(Runtime::measureTimings){
			cuEventRecord(ce_end, 0);
			float duration_host = now() - t_start;

			cuCtxSynchronize();

			float duration_device = 0.0f;
			cuEventElapsedTime(&duration_device, ce_start, ce_end);

			println("PaintUndoAction::compaction() timings: host: {:.3f} ms, device: {:.3f} ms", duration_host, duration_device);
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

struct SpherePaintAction : public InputAction{

	shared_ptr<PaintUndoAction> currentUndoAction = nullptr;
	static const int SPACE_WORLD  = 0;
	static const int SPACE_SCREEN = 1;

	inline static int space = SPACE_SCREEN;
	inline static float size_world = 1.0f;
	inline static float size_screen = 0.3f;
	inline static ImVec4 paintActionColor = ImVec4(1.0f, 0.0f, 0.0f, 0.5f);

	void beginUndoable(){
		auto editor = SplatEditor::instance;

		if(currentUndoAction){
			currentUndoAction->compaction();
			currentUndoAction = nullptr;
		}

		currentUndoAction = make_shared<PaintUndoAction>();
		
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

			PaintUndoAction::UndoData undodata;
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


	void start(){
		// auto editor = SplatEditor::instance;
	}

	void paintBrushStep(float radius){
		auto editor = SplatEditor::instance;

		if(editor->deviceState.hovered_depth < Infinity){
			paintBrushStep(editor->deviceState.hovered_pos, radius);
		}
	}

	void paintBrushStep(vec3 spherePos, float radius){
		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		{
			int numTicks = 1;

			editor->scene.process<SNSplats>([&](SNSplats* node) {

				if(!node->visible) return;
				if(node->locked) return;
				if(!editor->launchArgs.mouseEvents.isLeftDown) return;

				GaussianData& data = node->dmng.data;
				data.transform = node->transform_global;

				//editor->state.hasPendingPaintingTask = true;
				auto undodata = currentUndoAction->undodatas[node];
				CUdeviceptr cptr_stashedColors = undodata.cptr_stashedColors;
				
				vec4 color = {paintActionColor.x, paintActionColor.y, paintActionColor.z, paintActionColor.w};

				void* args[] = {&editor->launchArgs, &data, &cptr_stashedColors, &spherePos, &radius, &color, &numTicks};
				editor->prog_gaussians_editing->launch("kernel_paint_sphere", args, data.count);
			});
		}
	}

	void paintBrushSteps(vector<vec2> positions, float radius){
		// auto editor = SplatEditor::instance;
		// auto& settings = editor->settings;

		// if(editor->deviceState.hovered_depth < Infinity){
		// 	vec3 spherePos = editor->deviceState.hovered_pos;
			
		// 	int numTicks = 1;

		// 	editor->scene.process<SNSplats>([&](SNSplats* node) {

		// 		if(!node->visible) return;
		// 		if(node->locked) return;
		// 		if(!editor->launchArgs.mouseEvents.isLeftDown) return;

		// 		GaussianData& data = node->dmng.data;
		// 		data.transform = node->transform_global;

		// 		//editor->state.hasPendingPaintingTask = true;
		// 		auto undodata = currentUndoAction->undodatas[node];
		// 		CUdeviceptr cptr_stashedColors = undodata.cptr_stashedColors;
				
		// 		vec4 color = {paintActionColor.x, paintActionColor.y, paintActionColor.z, paintActionColor.w};

		// 		void* args[] = {&editor->launchArgs, &data, &cptr_stashedColors, &spherePos, &radius, &color, &numTicks};
		// 		editor->prog_gaussians_editing->launch("kernel_paint_sphere", args, data.count);
		// 	});
		// }
	}

	void update(){

		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		static bool wasLeftDown = false;
		bool isLeftDown = Runtime::mouseEvents.isLeftDown;
		bool leftReleased = wasLeftDown && !isLeftDown;
		bool leftPressed = wasLeftDown == false && isLeftDown == true;
		wasLeftDown = isLeftDown;

		float radius = 2.0f;
		if(space == SPACE_WORLD){
			radius = size_world;
		}else{
			radius = editor->deviceState.hovered_depth * size_screen / 2.0f;
		}
		const float MIN_STEPDISTANCE = 3.0f;
		// do a brush step every <stepdistance> pixels
		// float stepdistance = max(radius / 2.0f, MIN_STEPDISTANCE);
		float stepdistance = 50.0f;

		static vec2 lastMousePos = {0.0f, 0.0f};
		static float accumulatedDistance = 0.0f;

		float t_start = 0.0f;
		static CUevent ce_start = 0;
		static CUevent ce_end = 0;
		if(ce_start == 0){
			cuEventCreate(&ce_start, CU_EVENT_DEFAULT);
			cuEventCreate(&ce_end, CU_EVENT_DEFAULT);
		}

		if(Runtime::measureTimings){
			cuEventRecord(ce_start, 0);
			t_start = now();
		}

		// used in a workaround for issue #6
		// We will interpolate brush steps between the 3-dimensional hover positions
		static vec3 previousHoverPos;
		static float previousHoverDepth = Infinity;
		vec3 hoverPos = editor->deviceState.hovered_pos;
		float hoverDepth = editor->deviceState.hovered_depth;


		// static double lastTick = now();
		
		if(leftPressed){
			beginUndoable();
			// lastTick = now();
			lastMousePos = Runtime::mousePosition;
			accumulatedDistance = 0.0f;
		}else if (leftReleased) {
			endUndoable();
		}

		int numStepsPainted = 0;
		if(isLeftDown){

			if(Runtime::measureTimings){
				// During measurement, paint a step every frame so that we can nicely measure times
				paintBrushStep(radius);
			}else{
				// Otherwise, as a workaround to issue #6, paint a step whenever the mouse moves 5 pixels.

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
				vec2 a = lastMousePos;
				vec2 b = Runtime::mousePosition;
				vec2 ab = b - a;
				vec2 dir_ab = normalize(ab);
				vec2 O = a - accumulatedDistance * dir_ab;
				// float distance = length(b - a);
				float dist_Ob = length(b - O);

				if(leftPressed){
					vec2 brushPos = b;
					paintBrushStep(radius);
					numStepsPainted++;
				}else if(length(b - a) > 0.1f){

					// println("================================");
					// println("a: {:6.1f}, {:6.1f}", a.x, a.y);
					// println("b: {:6.1f}, {:6.1f}", b.x, b.y);
					// println("O: {:6.1f}, {:6.1f}", O.x, O.y);
					// println("accumulatedDistance: {:.1f}", accumulatedDistance);
					// println("dist_Ob:             {:.1f}", dist_Ob);
					// println("stepdistance:        {:.1f}", stepdistance);

					// Issue #6: hovered_pos is only available once per frame (at mouse pos b).
					// Interpolate the 3D brush positions between the previous frame's hovered_pos (a)
					// and the current one (b). If either is missing or there is a large depth jump
					// (e.g. at an object's silhouette), interpolation would paint into empty space,
					// so fall back to a single step at the current hovered_pos.
					bool canInterpolate =
						previousHoverDepth < Infinity &&
						hoverDepth < Infinity &&
						abs(hoverDepth - previousHoverDepth) < 0.2f * min(hoverDepth, previousHoverDepth);
					float len_ab = length(ab);

					vec2 lastBrushPos = O;
					int c = 0;
					for(float t = stepdistance; t < dist_Ob; t += stepdistance){
						vec2 brushPos = O + t * dir_ab;

						if(canInterpolate){
							// fraction along a->b at which this step lies
							float f = clamp((t - accumulatedDistance) / len_ab, 0.0f, 1.0f);
							vec3 spherePos = mix(previousHoverPos, hoverPos, f);
							paintBrushStep(spherePos, radius);
						}else if(c == 0){
							paintBrushStep(radius);
						}

						numStepsPainted++;

						lastBrushPos = brushPos;
						c++;
					}
					
					lastMousePos = b;
					accumulatedDistance = length(b - lastBrushPos);
				}
			}
		}

		// Issue #6: remember this frame's hovered position to interpolate brush steps next frame
		previousHoverPos = hoverPos;
		previousHoverDepth = hoverDepth;

		if(Runtime::measureTimings){
			cuEventRecord(ce_end, 0);
			float duration_host = now() - t_start;

			cuCtxSynchronize();

			float duration_device = 0.0f;
			cuEventElapsedTime(&duration_device, ce_start, ce_end);

			if(duration_device > 0.005f){
				println("SpherePaintAction::update() timings: host: {:.3f} ms, device: {:.3f} ms", duration_host, duration_device);
			}
		}

		if(editor->deviceState.hovered_depth < Infinity){
			auto sample = [&](float u){
				vec3 pos;
				pos.x = radius * cos(u);
				pos.y = radius * sin(u);
				pos.z = 0.0f;

				return pos;
			};

			int segments = 32;
			vec3 center = editor->deviceState.hovered_pos;
			for(int i = 0; i < segments; i++){

				float u  = 2.0f * 3.1415f * float(i) / float(segments);
				float u1 = 2.0f * 3.1415f * float(i + 1) / float(segments);

				vec3 p0 = sample(u);
				vec3 p1 = sample(u1);

				editor->drawLine(center + p0, center + p1, 0xffff0000);
				editor->drawLine(center + vec3{p0.z, p0.x, p0.y}, center + vec3{p1.z, p1.x, p1.y}, 0xff0000ff);
				editor->drawLine(center + vec3{p0.x, p0.z, p0.y}, center + vec3{p1.x, p1.z, p1.y}, 0xff00ff00);
			}
		}

		previousHoverPos = hoverPos;
		previousHoverDepth = hoverDepth;

	}

	void stop(){
		endUndoable();
	}

	void makeToolbarSettings(){

		auto editor = SplatEditor::instance;
		
		ImGui::Text("Space: ");

		ImGui::SameLine();
		ImGui::RadioButton("World", &space, SPACE_WORLD); 

		ImGui::SameLine();
		ImGui::RadioButton("Screen", &space, SPACE_SCREEN); 

		ImGui::SameLine();
		ImGui::Text("|");

		ImGui::SameLine();
		ImGui::Text("Size: ");

		if(space == SPACE_WORLD){

			ImGui::SameLine();
			ImGui::SetNextItemWidth(100.0f);
			ImGui::SliderFloat(" ", &size_world, 0.01f, 10.0f, "%.2f");

		}else if(space == SPACE_SCREEN){

			ImGui::SameLine();
			ImGui::SetNextItemWidth(100.0f);
			ImGui::SliderFloat(" ", &size_screen, 0.01f, 1.0f, "%.2f");
		}

		ImGui::SameLine();
		ImGui::Text("|");

		ImGui::SameLine();
		ImGui::Text("Color: ");

		ImGui::SameLine();
		static ImVec4 backup_color;
		// static ImVec4 color = ImVec4(114.0f / 255.0f, 144.0f / 255.0f, 154.0f / 255.0f, 200.0f / 255.0f);
		ImGuiColorEditFlags colorPickerFlags = 0; // = (hdr ? ImGuiColorEditFlags_HDR : 0) | (drag_and_drop ? 0 : ImGuiColorEditFlags_NoDragDrop) | (alpha_half_preview ? ImGuiColorEditFlags_AlphaPreviewHalf : (alpha_preview ? ImGuiColorEditFlags_AlphaPreview : 0)) | (options_menu ? 0 : ImGuiColorEditFlags_NoOptions);
		bool open_colorPicker = ImGui::ColorButton("Color", paintActionColor);

		if(open_colorPicker){
			ImGui::OpenPopup("ColorPicker");
			backup_color = paintActionColor;
		}

		if (ImGui::BeginPopup("ColorPicker"))
		{
			ImGui::Text("Color Picker");
			ImGui::Separator();
			ImGui::ColorPicker4("##picker", (float*)&paintActionColor, colorPickerFlags | ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoSmallPreview);
			ImGui::SameLine();

			ImGui::EndPopup();
		}

		ImGui::SameLine();
		ImGui::Text("|");

		ImGui::SameLine();
		ImGui::Text("Opacity: ");

		ImGui::SameLine();
		ImGui::SetNextItemWidth(100.0f);
		// ImGui::SliderFloat("##opacity", &editor->settings.brush.opacity, 0.01f, 1.0f);
		ImGui::SliderFloat("##opacity", &paintActionColor.w, 0.01f, 1.0f);

		ImGui::SameLine(); ImGui::Text("|");
		ImGui::SameLine(); ImGui::Text("Mode: ");
		ImGui::SameLine(); ImGui::RadioButton("Normal", &editor->settings.brushColorMode, BRUSHCOLORMODE_NORMAL);
		ImGui::SameLine(); ImGui::RadioButton("Hue&Saturation", &editor->settings.brushColorMode, BRUSHCOLORMODE_HUE_SATURATION);
	}

};