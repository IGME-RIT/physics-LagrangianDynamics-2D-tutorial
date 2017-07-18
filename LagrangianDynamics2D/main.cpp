/*
Title: LagrangianDynamics2D
File Name: main.cpp
Copyright © 2015
Original authors: Gabriel Ortega
Written under the supervision of David I. Schwartz, Ph.D., and
supported by a professional development seed grant from the B. Thomas
Golisano College of Computing & Information Sciences
(https://www.rit.edu/gccis) at the Rochester Institute of Technology.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Description:
This is a simulation of a pendulum swinging in 2D using Langragian equations of motion.
In Langragian dynamics, the acceleration is derived by determining the change in energy of the system.
The relevant equations are:

(Here I will use D to mean derivative and d to mean partial derivative, although readers should note that typically D is used to denote total derivative)

Kinetic Energy:
T = 1/2 * mass * velocity^2

Lagrangian Equation of motion for a single particle on a curve:
D/Dt(dT/dqDot) - dT/dq = F * dx,	where q is the parameter defining the curve to which the particle is constrained, qDot is the time derivative of q.
T is kinetic energy, F is external force (in this case gravity), and dx is the infinitesimal displacement.

What this means is that the change in energy of the system is equal to the Force exerted on the particle over the infinitesimal displacement.

Using these formulas along with polar coordinates we will be able to simulate the motion of a pendulum in 2D under the sole force of gravity.

Let the position of the pendulum be defined by:

x(t) = P + LN(Q), where Q is the angle of the pendulum arm.

Tangent to the path of motion is T = (cosQ, sinQ). Normal to the path of motion is N = (-sinQ, cosQ).

We solve this by treating x as a function of q, x(q) and q is a function of time q(t).
This leaves of with x(q(t)) which we can differentiate using the chain rule.
The velocity of the pendulum is		D/Dt(x(q(t))) = Dx/Dq * Dq/Dt = Dx/Dq * qDot
The q derivative of x is			Dx/Dq = L * DN/Dq = L * T
from which we can form				D/Dt(x(q(t))) = L * T * qDot

Plugging this into the formula for kinetic energy we have:
T = 1/2 * mass * (L * T * qDot)^2

Using partial derivatives we can derive:

dT/dq = 0
dT/qDot = mass * qDot * L^2

Now we can solve the equation of motion

D/Dt(dT/dqDot) - dT/dq = D/Dt(mass * qDot * L^2) - 0 = mass * qDotDot * L^2, where qDotDot is the second derivative of q... ACCELERATION!

The other side of the equation:

F * Dx = F * Dx/Dq = (mass * gravity * j) * (L * T) = -mass * gravity * L * sin(q), where j is the downward vector (0, -1, 0).

Since D/Dt(dT/dqDot) - dT/dq = F * dx we set both sides equal and solve for qDotDot (ACCELERATION):

(mass * qDotDot * L^2) = (-mass * gravity * L * sin(q))

qDotDot = -(gravity / L)sinq

Armed with acceleration, we can now integrate the q to get velocity for q and q itself which is the current angle of the pendulum arm.
That value can get plugged back into the position function.
*/

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\common.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <chrono>

#define PI															3.141592653589793238463
#define PENDULUM_RADIUS												1.0f
#define CIRCLE_VERTEX_COUNT											15							
#define PENDULUM_VERTEX_COUNT										17							// 15 vertices for circle + 2 vertices for the arm
#define GRAVITY														-0.00980665f

using namespace glm;

typedef std::chrono::high_resolution_clock::time_point				ClockTime;
typedef std::chrono::duration<double, std::milli>					Milliseconds;

GLFWwindow*	gWindow;
GLFWmonitor* gMonitor;
const GLFWvidmode* gVideoMode;
bool gShouldExit = 0;

GLuint gVertexBufferID = 0;
GLuint gVertexArrayID = 0;
GLuint gShaderID = 0;
GLuint gTransformID = 0;
GLuint gColorID = 0;

mat4 gProjection;
mat4 gView;
mat4 gModel;
mat4 gTransform;

vec3 gVertexBuffer[PENDULUM_VERTEX_COUNT];

vec3 gEyePosition = vec3(0.0f, 0.0f, 50.f);
vec3 gEyeDirection = vec3(0.0f, 0.0f, -1.0);
vec3 gEyeUp = vec3(0.0f, 1.0f, 0.0f);

void InitializePendulum();
void InitializeOpenGL();
void InitializeGeometry();
void InitializeShaders();
void InitializeProjectViewMatrices();
void BeginScene();
void BindGeometryAndShaders();
void UpdateScene(double millisecondsElapsed);
void RenderScene(double millisecondsElapsed);
void HandleInput();

void BuildCircle(vec3* vertexBuffer, const int& size, const float& radius);

struct Pendulum
{
	vec3 position;
	vec3 rotation;
	vec3 scale;
	vec3 velocity;
	vec3 pivot;
	float mass;
	float armAngle;
	float armLength;
} gPendulum;

// Main Loop
int main(int argc, int* argv[])
{
	// Program Structure
	InitializePendulum();
	InitializeOpenGL();
	InitializeGeometry();
	InitializeShaders();
	InitializeProjectViewMatrices();
	BeginScene();

	return 0;
}

void InitializePendulum()
{
	// Note that using Langrangian dynamics, we never actually calculate the velocity of the pendulum itself!
	gPendulum.position = { 0.0f, 0.0f, 0.0f };
	gPendulum.scale = { 1.0f, 1.0f, 1.0f };
	gPendulum.rotation = { 0.0f, 0.0f, 0.0f };
	gPendulum.pivot = { 0.0f, 20.0f, 0.0f };
	gPendulum.mass = 1.0f;
	gPendulum.armLength = 20.0f;
	gPendulum.armAngle = (float)PI * 0.25f;				// Start at 90 degrees
}

void InitializeOpenGL()
{
	// Graphics API setup.
	int glfwSuccess = glfwInit();
	if (!glfwSuccess) {
		exit(1);
	}

	// Create Window
	gMonitor = glfwGetPrimaryMonitor();
	gVideoMode = glfwGetVideoMode(gMonitor);

	//GLFWwindow* window = glfwCreateWindow(videoMode->width, videoMode->height, "Sphere", NULL, NULL);
	gWindow = glfwCreateWindow(1600, 1200, "Langrangian Dynamics 2D", NULL, NULL);

	if (!gWindow) {
		glfwTerminate();
	}

	glfwMakeContextCurrent(gWindow);
	glewInit();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
}

void InitializeGeometry()
{
	// Calculate Vertices for a circle.
	BuildCircle(gVertexBuffer, CIRCLE_VERTEX_COUNT, PENDULUM_RADIUS);

	// Add two points on the end for the arm.
	gVertexBuffer[PENDULUM_VERTEX_COUNT - 2] = gPendulum.pivot;
	gVertexBuffer[PENDULUM_VERTEX_COUNT - 1] = gPendulum.position;

	// Bind vertex data to OpenGL
	glGenBuffers(1, &gVertexBufferID);
	glBindBuffer(GL_ARRAY_BUFFER, gVertexBufferID); // OpenGL.GL_Array_Buffer = buffer with ID(vertexBufferID)
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * PENDULUM_VERTEX_COUNT, gVertexBuffer, GL_STATIC_DRAW);
}

void InitializeShaders()
{
	// Extremely simple vertex and fragment shaders
	const char* vertex_shader =
		"#version 400\n"
		"uniform mat4 transform;"
		"in vec3 vp;"
		"void main () {"
		"  gl_Position = transform * vec4 (vp, 1.0);"
		"}";

	const char* fragment_shader =
		"#version 400\n"
		"uniform vec4 color;"
		"out vec4 frag_colour;"
		"void main () {"
		"  frag_colour = color;"
		"}";

	GLuint vShaderID = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vShaderID, 1, &vertex_shader, NULL);
	glCompileShader(vShaderID);

	GLuint fShaderID = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fShaderID, 1, &fragment_shader, NULL);
	glCompileShader(fShaderID);

	gShaderID = glCreateProgram();
	glAttachShader(gShaderID, vShaderID);
	glAttachShader(gShaderID, fShaderID);
	glLinkProgram(gShaderID);

	glBindBuffer(GL_ARRAY_BUFFER, gVertexBufferID);

	glGenVertexArrays(1, &gVertexArrayID);
	glBindVertexArray(gVertexArrayID);

	GLuint attributeID = glGetAttribLocation(gShaderID, "vp");
	glVertexAttribPointer(attributeID, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(attributeID);

	gTransformID = glGetUniformLocation(gShaderID, "transform");
	gColorID = glGetUniformLocation(gShaderID, "color");
}

void InitializeProjectViewMatrices()
{
	// Camera is static so only calculate projection and view matrix once.
	gProjection = perspective(45.0f, (float)gVideoMode->width / (float)gVideoMode->height, 0.1f, 100.0f);
	gView = lookAt(gEyePosition, gEyePosition + gEyeDirection, gEyeUp);
}

void BeginScene()
{
	// Loop setup. 
	ClockTime currentTime = std::chrono::high_resolution_clock::now();
	while (!gShouldExit)
	{
		ClockTime systemTime = std::chrono::high_resolution_clock::now();
		double deltaTime = Milliseconds(systemTime - currentTime).count() + DBL_EPSILON;
		currentTime = systemTime;

		UpdateScene(deltaTime);
		RenderScene(deltaTime);
		glfwSwapBuffers(gWindow);
	}
}

void UpdateScene(double millisecondsElapsed)
{
	if (millisecondsElapsed > 16.67f)
	{
		millisecondsElapsed = 16.67f;
	}

	double accumulator = millisecondsElapsed;
	double deltaTime = millisecondsElapsed / 5.0f;
	double time = 0.0f;

	while (accumulator >= time)
	{
		float q = gPendulum.armAngle;
		static float qDot = 0.0f;

		// Calculate the acceleration of the curve
		float qDotDot = ((float)GRAVITY / gPendulum.armLength) * sinf(q);
		qDot += qDotDot * (float)deltaTime;
		q += qDot * (float)deltaTime;
		gPendulum.armAngle = q;

		vec3 normal = { sinf(q), -cosf(q), 0.0f };
		vec3 tangent = { cosf(q), sinf(q), 0.0f };

		// Plug new angle into  the position funtion.
		gPendulum.position = gPendulum.armLength * normal;

		// This is simply here so the pendulum arm looks proper. This has nothing to do with the physics.
		gPendulum.rotation.z = q;

		time += deltaTime;
		accumulator -= deltaTime;
	}

	// Update Graphics Data 
	mat4 identity = mat4(1.0f);
	gModel = scale(rotate(translate(identity, gPendulum.position), gPendulum.rotation.z, vec3(0.0f, 0.0f, 1.0f)), gPendulum.scale);
	gTransform = gProjection * gView * gModel;

	// Update Title
	char title[100];
	float fps = 1000.0f / (float)millisecondsElapsed;
	sprintf_s(title, "Langrangian Dynamics 2D FPS: %f", fps);
	glfwSetWindowTitle(gWindow, title);
}

void RenderScene(double millisecondsElapsed)
{
	// Clear buffers. Set shader.
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glUseProgram(gShaderID);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	// Bind vertex layout.
	glBindVertexArray(gVertexArrayID);

	// Bind shader data.
	glUniformMatrix4fv(gTransformID, 1, GL_FALSE, &gTransform[0][0]);

	// Draw pendulum and arm.
	glDrawArrays(GL_TRIANGLE_FAN, 0, CIRCLE_VERTEX_COUNT);
	glDrawArrays(GL_LINES, CIRCLE_VERTEX_COUNT, 2);
}

void BuildCircle(vec3* vertexBuffer, const int& size, const float& radius)
{
	float radianStep = (float)((2.0 * PI) / size);
	for (int i = 0; i < size; i++) {
		float radians = i * radianStep;
		vertexBuffer[i] = vec3(radius * cosf(radians), radius * sinf(radians), 0.0f);
	}
}
