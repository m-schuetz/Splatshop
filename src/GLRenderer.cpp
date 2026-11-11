
#include <filesystem>
#include <print>

#include "GLRenderer.h"
#include "Runtime.h"

namespace fs = std::filesystem;

using std::print;


static void APIENTRY debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {

	if (
		severity == GL_DEBUG_SEVERITY_NOTIFICATION 
		|| severity == GL_DEBUG_SEVERITY_LOW 
		|| severity == GL_DEBUG_SEVERITY_MEDIUM
		) {

		// println("OPENGL DEBUG CALLBACK: {}", message);

		return;
	}

	cout << "OPENGL DEBUG CALLBACK: " << message << endl;
}

void error_callback(int error, const char* description){
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){

	// cout << "key: " << key << ", scancode: " << scancode << ", action: " << action << ", mods: " << mods << endl;
	// println("key: {}, scancode: {}, action: {}", key, scancode, action);

	if (key < 0 || key >= Runtime::keyStates.size()) {
		println("could not handle key input - ignoring input.");
		return;
	}

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}

	// const char* keyname = glfwGetKeyName(key, scancode);
	// println("keyname: {}", keyname);

	Runtime::keyStates[key] = action;
	Runtime::mods = mods;

	Runtime::frame_keys.push_back(key);
	Runtime::frame_scancodes.push_back(scancode);
	Runtime::frame_actions.push_back(action);
	Runtime::frame_mods.push_back(mods);
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse){
		return;
	}
	
	Runtime::mousePosition = {xpos, ypos};

	int width, height;
	glfwGetWindowSize(window, &width, &height);
	// Runtime::controls->onMouseMove(xpos, height - ypos);
	Runtime::mouseEvents.onMouseMove(xpos, height - ypos);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse){
		return;
	}

	// Runtime::controls->onMouseScroll(xoffset, yoffset);
	Runtime::mouseEvents.onMouseScroll(xoffset, yoffset);
}


void drop_callback(GLFWwindow* window, int count, const char **paths){
	for(int i = 0; i < count; i++){
		cout << paths[i] << endl;
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods){

	// if(action != 0)
	// {
	// 	// GLFW_PRESS
	// 	GLFW_RELEASE
	// 	println("{}, {}, {}", button, action, mods);
	// }
	

	// cout << "start button: " << button << ", action: " << action << ", mods: " << mods << endl;

	// Only filter presses. WantCaptureMouse reflects the previous frame, so a press may reach
	// the editor while the matching release is swallowed by ImGui (e.g. when clicking into the
	// window while it was out of focus), leaving the button stuck down. See issue #9.
	// Releases are therefore always forwarded.
	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse && action == GLFW_PRESS){
		return;
	}

	// cout << "end button: " << button << ", action: " << action << ", mods: " << mods << endl;

	


	if(action == 1){
		Runtime::mouseButtons = Runtime::mouseButtons | (1 << button);
	}else if(action == 0){
		uint32_t mask = ~(1 << button);
		Runtime::mouseButtons = Runtime::mouseButtons & mask;
	}

	Runtime::controls->onMouseButton(button, action, mods);
	Runtime::mouseEvents.onMouseButton(button, action, mods);
}

void GLRenderer::init(){
	
	camera = make_shared<Camera>();

	glfwSetErrorCallback(error_callback);

	if (!glfwInit()) {
		// Initialization failed
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_DECORATED, true);

	int numMonitors;
	GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);

	cout << "<create windows>" << endl;
	// if (numMonitors > 1) 
	// {
	// 	const GLFWvidmode * modeLeft = glfwGetVideoMode(monitors[0]);
	// 	const GLFWvidmode * modeRight = glfwGetVideoMode(monitors[1]);

	// 	// int width = modeRight->width;
	// 	// int height = modeRight->height - 300;
	// 	int width = 1920;
	// 	int height = 1080;
	// 	window = glfwCreateWindow(
	// 		width, height, 
	// 		"Splat Editor", nullptr, nullptr);

	// 	if (!window) {
	// 		glfwTerminate();
	// 		exit(EXIT_FAILURE);
	// 	}

	// 	int xpos;
	// 	int ypos;
	// 	glfwGetMonitorPos(monitors[1], &xpos, &ypos);
	// 	glfwSetWindowPos(window, xpos, ypos);
	// } 
	// else
	{
		const GLFWvidmode * mode = glfwGetVideoMode(monitors[0]);

		// window = glfwCreateWindow(mode->width - 100, mode->height - 100, "Splat Editor", nullptr, nullptr);

		int width = 1920;
		int height = 1080;
		window = glfwCreateWindow(width, height, "Splat Editor", nullptr, nullptr);

		if (!window) {
			glfwTerminate();
			exit(EXIT_FAILURE);
		}

		glfwSetWindowPos(window, 50, 50);
	}

	cout << "<set input callbacks>" << endl;
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);

	// Release any mouse buttons still held when the window loses focus,
	// otherwise the release event may never arrive. See issue #9.
	glfwSetWindowFocusCallback(window, [](GLFWwindow* window, int focused){
		if(focused) return;

		for(int button = 0; button < 3; button++){
			if(Runtime::mouseButtons & (1 << button)){
				mouse_button_callback(window, button, GLFW_RELEASE, 0);
			}
		}
	});
	//glfwSetDropCallback(window, drop_callback);

	glfwSetDropCallback(window, [](GLFWwindow*, int count, const char **paths){

		vector<string> files;
		for(int i = 0; i < count; i++){
			string file = paths[i];
			files.push_back(file);
		}

		for(auto &listener : GLRenderer::fileDropListeners){
			listener(files);
		}
	});

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "glew error: %s\n", glewGetErrorString(err));
	}

	cout << "<glewInit done> " << "(" << now() << ")" << endl;

	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_HIGH, 0, NULL, GL_TRUE);
	glDebugMessageCallback(debugCallback, NULL);

	glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);

	{ // SETUP IMGUI
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImPlot::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init("#version 450");
		ImGui::StyleColorsDark();
	}

	view.framebuffer = GLRenderer::createFramebuffer(128, 128);
}

shared_ptr<GLTexture> GLRenderer::createTexture(int width, int height, GLuint colorType) {

	auto texture = GLTexture::create(width, height, colorType);

	return texture;
}

shared_ptr<Framebuffer> GLRenderer::createFramebuffer(int width, int height) {

	auto framebuffer = Framebuffer::create();

	GLenum status = glCheckNamedFramebufferStatus(framebuffer->handle, GL_FRAMEBUFFER);

	if (status != GL_FRAMEBUFFER_COMPLETE) {
		cout << "framebuffer incomplete" << endl;
	}

	framebuffer->setSize(width, height);

	return framebuffer;
}

void GLRenderer::loop(function<void(void)> update, function<void(void)> render){

	int fpsCounter = 0;
	double start = now();
	double tPrevious = start;
	double tPreviousFPSMeasure = start;

	vector<float> frameTimes(1000, 0);

	while (!glfwWindowShouldClose(window)){

		// TIMING
		// double timeSinceLastFrame;
		{
			double tCurrent = now();
			timeSinceLastFrame = tCurrent - tPrevious;
			tPrevious = tCurrent;

			double timeSinceLastFPSMeasure = tCurrent - tPreviousFPSMeasure;

			if(timeSinceLastFPSMeasure >= 0.3){
				GLRenderer::fps = double(fpsCounter) / timeSinceLastFPSMeasure;

				tPreviousFPSMeasure = tCurrent;
				fpsCounter = 0;
			}
			frameTimes[frameCount % frameTimes.size()] = timeSinceLastFrame;
		}
		

		// WINDOW
		int width, height;
		glfwGetWindowSize(window, &width, &height);
		camera->setSize(width, height);
		GLRenderer::width = width;
		GLRenderer::height = height;

		EventQueue::instance->process();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, GLRenderer::width, GLRenderer::height);

		glBindFramebuffer(GL_FRAMEBUFFER, view.framebuffer->handle);
		glViewport(0, 0, GLRenderer::width, GLRenderer::height);


		{ 
			Runtime::controls->update();

			camera->world = Runtime::controls->world;
			camera->position = camera->world * dvec4(0.0, 0.0, 0.0, 1.0);
		}

		
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();

		{ // UPDATE & RENDER
			camera->update();
			update();
			camera->update();
		}


		// ImGui::NewFrame();
		// ImGuizmo::BeginFrame();

		render();

		Runtime::frame_keys.clear();
		Runtime::frame_scancodes.clear();
		Runtime::frame_actions.clear();
		Runtime::frame_mods.clear();

		// ImGui::Render();
		// ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		auto source = view.framebuffer;
		glBlitNamedFramebuffer(
			source->handle, 0,
			0, 0, source->width, source->height,
			0, 0, 0 + source->width, 0 + source->height,
			GL_COLOR_BUFFER_BIT, GL_LINEAR);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, GLRenderer::width, GLRenderer::height);

		// Runtime::mouseEvents.clear();


		glfwSwapBuffers(window);
		glfwPollEvents();

		fpsCounter++;
		frameCount++;
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}

shared_ptr<Framebuffer> Framebuffer::create() {

	auto fbo = make_shared<Framebuffer>();

	glCreateFramebuffers(1, &fbo->handle);

	{ // COLOR ATTACHMENT 0

		auto texture = GLRenderer::createTexture(fbo->width, fbo->height, GL_RGBA8);
		fbo->colorAttachments.push_back(texture);

		glNamedFramebufferTexture(fbo->handle, GL_COLOR_ATTACHMENT0, texture->handle, 0);
	}

	{ // DEPTH ATTACHMENT

		auto texture = GLRenderer::createTexture(fbo->width, fbo->height, GL_DEPTH_COMPONENT32F);
		fbo->depth = texture;

		glNamedFramebufferTexture(fbo->handle, GL_DEPTH_ATTACHMENT, texture->handle, 0);
	}

	fbo->setSize(128, 128);

	return fbo;
}

shared_ptr<GLTexture> GLTexture::create(int width, int height, GLuint colorType){

	GLuint handle;
	glCreateTextures(GL_TEXTURE_2D, 1, &handle);

	auto texture = make_shared<GLTexture>();
	texture->handle = handle;
	texture->colorType = colorType;

	texture->setSize(width, height);

	texture->ID = GLTexture::idcounter;
	idcounter++;

	return texture;
}

void GLTexture::setSize(int width, int height) {

	bool needsResize = this->width != width || this->height != height;

	if (needsResize) {

		glDeleteTextures(1, &this->handle);
		glCreateTextures(GL_TEXTURE_2D, 1, &this->handle);

		glTextureParameteri(this->handle, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(this->handle, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		// glTextureParameteri(this->handle, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		// glTextureParameteri(this->handle, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTextureParameteri(this->handle, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTextureParameteri(this->handle, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glTextureStorage2D(this->handle, 1, this->colorType, width, height);

		this->width = width;
		this->height = height;

		version++;
	}

}