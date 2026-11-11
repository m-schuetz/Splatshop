#include <deque>
#include "imutils.h"


void makeKernels(){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	auto& state = editor->state;

	if(settings.showKernelInfos){

		ImVec2 kernelWindowSize = {800, 600};
		ImGui::SetNextWindowPos({
			(GLRenderer::width - kernelWindowSize.x) / 2, 
			(GLRenderer::height - kernelWindowSize.y) / 2, }, 
			ImGuiCond_Once);
		ImGui::SetNextWindowSize(kernelWindowSize, ImGuiCond_Once);

		if(ImGui::Begin("Kernels")){

			static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;


			auto printKernels = [&](string label, CudaModularProgram* program){

				string strlabel = format("## {}", label);
				ImGui::Text("===============================");
				ImGui::Text(strlabel.c_str());
				ImGui::Text("===============================");
				
				if(ImGui::BeginTable("Kernels##listOfKernels", 5, flags))
				{
					ImGui::TableSetupColumn("Name",       ImGuiTableColumnFlags_WidthStretch, 3.0f);
					ImGui::TableSetupColumn("registers",  ImGuiTableColumnFlags_WidthStretch, 1.0f);
					ImGui::TableSetupColumn("shared mem", ImGuiTableColumnFlags_WidthStretch, 1.0f);
					ImGui::TableSetupColumn("max threads/block", ImGuiTableColumnFlags_WidthStretch, 1.0f);
					ImGui::TableSetupColumn("blocks(256t)/SM", ImGuiTableColumnFlags_WidthStretch, 1.0f);

					ImGui::TableHeadersRow();

					for(auto [name, function] : program->kernels){
						ImGui::TableNextRow();
				
						int maxThreadsPerBlock = 0;
						int registersPerThread;
						int sharedMemory;
						// With lazy module loading (default since CUDA 12.2), functions must be loaded before querying attributes
						CURuntime::check(cuFuncLoad(function), false);
						CURuntime::check(cuFuncGetAttribute(&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function), false);
						CURuntime::check(cuFuncGetAttribute(&registersPerThread, CU_FUNC_ATTRIBUTE_NUM_REGS, function), false);
						CURuntime::check(cuFuncGetAttribute(&sharedMemory, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function), false);

						

						int numBlocksPerSM;
						cuOccupancyMaxActiveBlocksPerMultiprocessor (&numBlocksPerSM, function, 256, 0);

						string strThreadsPerBlock = format("{}", maxThreadsPerBlock);
						if(maxThreadsPerBlock == 0) strThreadsPerBlock = "?";
						string strRegisters = format("{}", registersPerThread);
						string strSharedMem = format(getSaneLocale(), "{:L}", sharedMemory);
						string strBlocksPerSM = format(getSaneLocale(), "{:L}", numBlocksPerSM);

						ImGui::TableNextColumn();
						ImGui::Text(name.c_str());

						ImGui::TableNextColumn();
						ImUtils::alignRight(strRegisters);
						ImGui::Text(strRegisters.c_str());

						ImGui::TableNextColumn();
						ImUtils::alignRight(strSharedMem);
						ImGui::Text(strSharedMem.c_str());

						ImGui::TableNextColumn();
						ImUtils::alignRight(strThreadsPerBlock);
						ImGui::Text(strThreadsPerBlock.c_str());

						ImGui::TableNextColumn();
						ImUtils::alignRight(strBlocksPerSM);
						ImGui::Text(strBlocksPerSM.c_str());
					}

					ImGui::EndTable();
				}

				// for(auto [name, function] : program->kernels){
				
				// 	int registersPerThread;
				// 	int sharedMemory;
				// 	cuFuncGetAttribute(&registersPerThread, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
				// 	cuFuncGetAttribute(&sharedMemory, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function);

				// 	string strKernel = format("\"{}\"", name);
				// 	string strRegisters = format("    registers per thread  {:10}", registersPerThread);
				// 	string strSharedMem = format("    shared memory         {:10}", sharedMemory);

				// 	ImGui::Text(" ");
				// 	ImGui::Text(strKernel.c_str());
				// 	ImGui::Text(strRegisters.c_str());
				// 	ImGui::Text(strSharedMem.c_str());
				// }
			};

			printKernels("GAUSSIAN", editor->prog_gaussians_rendering);
			printKernels("GAUSSIAN", editor->prog_gaussians_editing);
			printKernels("RADIX SORT (GPUSorting)", GPUSorting::prog_radix);
			printKernels("POINTS", editor->prog_points);
			printKernels("HELPERS", editor->prog_helpers);
		}

		ImGui::End();
	}

}

void makeMemory(){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	auto& state = editor->state;

	if(settings.showMemoryInfos){

		// auto windowSize = ImGui::GetWindowSize();
		ImVec2 windowSize = {800, 600};
		ImGui::SetNextWindowPos({
			(GLRenderer::width - windowSize.x) / 2, 
			(GLRenderer::height - windowSize.y) / 2, }, 
			ImGuiCond_Once);
		ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);

		if(ImGui::Begin("Memory")){

			static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;

			ImGui::Text("List of allocations made via CURuntime::alloc and allocVirtual.");
			ImGui::Text(" ");

			ImGui::Text("===============================");
			ImGui::Text("## MEMORY ALLOCATIONS");
			ImGui::Text("===============================");

			if(ImGui::BeginTable("Memory", 2, flags)){

				ImGui::TableSetupColumn("Label",             ImGuiTableColumnFlags_WidthStretch, 3.0f);
				ImGui::TableSetupColumn("Allocated Memory",  ImGuiTableColumnFlags_WidthStretch, 1.0f);

				ImGui::TableHeadersRow();

				int64_t sum = 0;
				for(auto allocation : CURuntime::allocations){
					
					ImGui::TableNextRow();

					ImGui::TableNextColumn();
					ImGui::Text(allocation.label.c_str());

					ImGui::TableNextColumn();
					string strMemory = format(getSaneLocale(), "{:L}", allocation.size);
					ImUtils::alignRight(strMemory);
					ImGui::Text(strMemory.c_str());

					sum += allocation.size;
				}

				{
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("-----------------------");
					ImGui::TableNextColumn();
					ImGui::Text(" ");

					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("Total");
					ImGui::TableNextColumn();
					string strTotal = format(getSaneLocale(), "{:L}", sum);
					ImUtils::alignRight(strTotal);
					ImGui::Text(strTotal.c_str());
				}

				ImGui::EndTable();
			}

			ImGui::Text("===============================");
			ImGui::Text("## VIRTUAL MEMORY ALLOCATIONS");
			ImGui::Text("===============================");

			if(ImGui::BeginTable("Memory", 2, flags)){

				ImGui::TableSetupColumn("Label",             ImGuiTableColumnFlags_WidthStretch, 3.0f);
				ImGui::TableSetupColumn("allocated/comitted memory",  ImGuiTableColumnFlags_WidthStretch, 1.0f);

				ImGui::TableHeadersRow();

				int64_t sum = 0;
				for(auto allocation : CURuntime::allocations_virtual){
					
					ImGui::TableNextRow();

					ImGui::TableNextColumn();
					ImGui::Text(allocation.label.c_str());

					ImGui::TableNextColumn();
					string strMemory = format(getSaneLocale(), "{:L}", allocation.memory->comitted);
					ImUtils::alignRight(strMemory);
					ImGui::Text(strMemory.c_str());

					sum += allocation.memory->comitted;
				}

				{
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("-----------------------");
					ImGui::TableNextColumn();
					ImGui::Text(" ");

					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("Total");
					ImGui::TableNextColumn();
					string strTotal = format(getSaneLocale(), "{:L}", sum);
					ImUtils::alignRight(strTotal);
					ImGui::Text(strTotal.c_str());
				}

				ImGui::EndTable();
			}
		}

		ImGui::End();
	}

}

void makeTimings(){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	auto& state = editor->state;

	if(settings.showTimingInfos){

		ImVec2 windowSize = {800, 700};
		ImGui::SetNextWindowPos({
			GLRenderer::width - windowSize.x - 10,
			(GLRenderer::height - windowSize.y) / 2, }, 
			ImGuiCond_Once);
		ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);

		if(ImGui::Begin("Timings")){

			constexpr int HISTORY_SIZE = 50;
			static unordered_map<string, deque<CudaModularProgram::Timing>> history;

			static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;

			string txt = format("Displaying median kernel execution duration over {} frames", HISTORY_SIZE);
			ImGui::Text(txt.c_str());
			ImGui::Text("Note: If there are more than 1 invocations of a single kernel in a frame, each will display the same median.");
			ImGui::Text("Thus, timings are only useful for the kernels that are invoked once per frame. ");

			// static bool enableUpdateTimings = true;
			ImGui::Checkbox("update", &Runtime::measureTimings);
			
			if (ImGui::BeginTable("Timings", 3, flags)){

				// HEADER
				ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthStretch, 4.0f);
				ImGui::TableSetupColumn("Milliseconds", ImGuiTableColumnFlags_WidthStretch, 1.0f);
				ImGui::TableSetupColumn("Sum", ImGuiTableColumnFlags_WidthStretch, 1.0f);
				ImGui::TableHeadersRow();

				// auto timings = Runtime::timings;

				auto timeToColor = [](float duration){
					
					// https://colorbrewer2.org/#type=diverging&scheme=RdYlGn&n=10
					uint32_t color = 0xffffffff; 
					if(duration > 10.0)      {color = IM_COL32(165,  0, 38, 255);}
					else if(duration > 5.0)  {color = IM_COL32(215, 48, 39, 255);}
					else if(duration > 1.0)  {color = IM_COL32(244,109, 67, 255);}
					else if(duration > 0.5)  {color = IM_COL32(253,174, 97, 255);}
					else if(duration > 0.1)  {color = IM_COL32(254,224,139, 255);}
					else if(duration > 0.0)  {color = IM_COL32(217,239,139, 255);}

					return color;
				};

				// static Timings runtimeTimings;
				// static vector<CudaModularProgram::Timing> timings;
				// static double lastUpdateAt = now();
				// constexpr double updateEveryXSecond = 0.1;
				// double timeSinceLastTiming = now() - lastUpdateAt;

				if(Runtime::measureTimings){
					// ADD NEW TIMINGS TO HISTORY

					// unordered_map<string, int> launchCounter;

					for(CudaModularProgram::Timing timing : CudaModularProgram::timings){
						
						timing.frame = GLRenderer::frameCount;
						// timing.launchIndex = launchCounter[timing.kernelName];

						// launchCounter[timing.kernelName] += 1;

						history[timing.kernelName].push_back(timing);
						if(history[timing.kernelName].size() > HISTORY_SIZE){
							history[timing.kernelName].pop_front();
						}
					}
				}

				auto computeMedian = [&](string kernelName, int launchIndex = 0){
					// deque<CudaModularProgram::Timing> entries = history[kernelName];

					vector<CudaModularProgram::Timing> entries;
					for(auto timing : history[kernelName]){
						if(timing.launchIndex == launchIndex){
							entries.push_back(timing);
						}
					}

					std::sort(entries.begin(), entries.end(), [](CudaModularProgram::Timing a, CudaModularProgram::Timing b){
						return a.device_duration < b.device_duration;
					});

					return entries[entries.size() / 2];
				};


				// if(Runtime::measureTimings && timeSinceLastTiming > updateEveryXSecond){
				// 	timings = CudaModularProgram::timings;
				// 	runtimeTimings = Runtime::timings;
				// 	lastUpdateAt = now();
				// }

				for(auto [label, list] : Runtime::timings.entries){

					float duration = Runtime::timings.getAverage(label);

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted(label.c_str());

					ImGui::TableSetColumnIndex(1);
					string strTime = format("{:6.3f}", duration);
					// ImGui::TextUnformatted(strTime.c_str());

					ImU32 color = timeToColor(duration); 
					ImGui::PushStyleColor(ImGuiCol_Text, color);
					ImGui::Text(strTime.c_str());
					ImGui::PopStyleColor();

					ImGui::TableSetColumnIndex(2);
					ImGui::Text(" ");

				}
				
				float sum = 0.0f;
				for(auto timing : CudaModularProgram::timings){

					auto medianTiming = computeMedian(timing.kernelName, timing.launchIndex);

					float duration = medianTiming.device_duration;
					sum += duration;

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted(timing.kernelName.c_str());

					ImGui::TableSetColumnIndex(1);
					string strTime = format("{:6.3f}", duration);

					ImU32 color = timeToColor(duration); 
					ImGui::PushStyleColor(ImGuiCol_Text, color);
					ImGui::Text(strTime.c_str());
					ImGui::PopStyleColor();

					ImGui::TableSetColumnIndex(2);
					string strLaunches = format("{:.3f}", sum);
					// string strLaunches = " ";
					ImGui::Text(strLaunches.c_str());

				}

				ImGui::EndTable();
			}
			
		}

		ImGui::End();
	}
	// Runtime::measureTimings = settings.showTimingInfos;

}

void SplatEditor::makeDevGUI(){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	auto& state = editor->state;

	int margin = 25;

	makeKernels();
	makeMemory();
	makeTimings();

	if(!editor->settings.showDevStuff) return;

	ImGui::SetNextWindowPos(ImVec2(0, 57 + 16 + 17 + 14));
	ImGui::SetNextWindowSize(ImVec2(490, GLRenderer::height - (57 + 16)));
	ImGui::SetNextWindowBgAlpha(1.0f);

	ImGui::Begin("Dev & Debug");

	if(ImGui::Button("Kernel Infos")){
		settings.showKernelInfos = !settings.showKernelInfos;
	}

	ImGui::SameLine();
	if(ImGui::Button("Memory")){
		settings.showMemoryInfos = !settings.showMemoryInfos;
	}

	ImGui::SameLine();
	if(ImGui::Button("Timings")){
		settings.showTimingInfos = !settings.showTimingInfos;
	}


	ImGui::Text("Method: ");
	// static int e = 0;
	ImGui::RadioButton("Original", &state.dbg_method, 0); ImGui::SameLine();
	ImGui::RadioButton("TileSubsets", &state.dbg_method, 1); ImGui::SameLine();
	// ImGui::RadioButton("Warpwise", &state.dbg_method, 2); ImGui::SameLine();
	ImGui::RadioButton("FetchFilter", &state.dbg_method, 3); 

	ImGui::SliderFloat("Dbg Factor", &settings.dbg_factor, 1.0f, 100.0f, "%.1f");


	makePerf();
	makeSettings();

	ImGui::End();

}