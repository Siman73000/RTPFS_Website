// main.cpp
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengles2.h>
#include <emscripten.h>

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <random>

// -------------------------------
// Configuration
// -------------------------------
const int NUM_PARTICLES = 10000;  // Particle count
const float dt = 0.016f;          // Time step (~60 FPS)
const float gravity[3] = {0.0f, -9.81f, 0.0f};  // Gravity force
const float damping = 0.99f;      // Velocity damping

// For collision resolution:
const float cellSize = 0.1f;     // Grid cell size
const float minDistance = 0.0999f;
const float stiffness = 1200.0f; // Repulsion constant

// -------------------------------
// Particle structure
// -------------------------------
struct Particle {
    float pos[3];
    float vel[3];
};

// Global container for particles
std::vector<Particle> particles;

// -------------------------------
// OpenGL globals
// -------------------------------
SDL_Window* window = nullptr;
SDL_GLContext glContext = nullptr;
GLuint program = 0;
GLuint vbo = 0;
GLint attrib_position = 0;
GLint uniform_mvp = 0;

// The Model-View-Projection matrix.
float mvp[16];

// Frame counter for debugging output
int frameCount = 0;

// -------------------------------
// Matrix math utility functions
// -------------------------------

// Multiply two 4x4 matrices: result = a * b.
void mat4_mult(const float a[16], const float b[16], float result[16]) {
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col) {
            result[row * 4 + col] = 0.0f;
            for (int k = 0; k < 4; ++k)
                result[row * 4 + col] += a[row * 4 + k] * b[k * 4 + col];
        }
}

// Build a perspective projection matrix.
void perspective(float fovy, float aspect, float near, float far, float out[16]) {
    float f = 1.0f / tanf(fovy * 0.5f);
    out[0] = f / aspect;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    
    out[4] = 0;
    out[5] = f;
    out[6] = 0;
    out[7] = 0;
    
    out[8] = 0;
    out[9] = 0;
    out[10] = (far + near) / (near - far);
    out[11] = -1;
    
    out[12] = 0;
    out[13] = 0;
    out[14] = (2 * far * near) / (near - far);
    out[15] = 0;
}

// Build a simple look-at view matrix.
void lookAt(const float eye[3], const float center[3], const float up[3], float out[16]) {
    float f[3] = { center[0] - eye[0],
                   center[1] - eye[1],
                   center[2] - eye[2] };
    float f_norm = sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    f[0] /= f_norm; f[1] /= f_norm; f[2] /= f_norm;
    
    float s[3] = { f[1]*up[2] - f[2]*up[1],
                   f[2]*up[0] - f[0]*up[2],
                   f[0]*up[1] - f[1]*up[0] };
    float s_norm = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    s[0] /= s_norm; s[1] /= s_norm; s[2] /= s_norm;
    
    float u[3] = { s[1]*f[2] - s[2]*f[1],
                   s[2]*f[0] - s[0]*f[2],
                   s[0]*f[1] - s[1]*f[0] };
    
    // Column-major order.
    out[0]  = s[0];
    out[1]  = u[0];
    out[2]  = -f[0];
    out[3]  = 0;
    
    out[4]  = s[1];
    out[5]  = u[1];
    out[6]  = -f[1];
    out[7]  = 0;
    
    out[8]  = s[2];
    out[9]  = u[2];
    out[10] = -f[2];
    out[11] = 0;
    
    out[12] = - (s[0]*eye[0] + s[1]*eye[1] + s[2]*eye[2]);
    out[13] = - (u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2]);
    out[14] = f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2];
    out[15] = 1;
}

// Compute and store the MVP matrix.
void computeMVP() {
    // Create a perspective projection matrix.
    float proj[16];
    float fov = 45.0f * (3.14159265f / 180.0f); // Convert FOV to radians.
    perspective(fov, 800.0f / 600.0f, 0.1f, 100.0f, proj);

    // Create a view matrix.
    // Set the camera further back (eye position at (0, 0, 5)) to see the particles.
    float view[16];
    const float eye[3]    = {0.0f, 0.0f, 5.0f};  // Moved from 3.0f to 5.0f on Z axis.
    const float center[3] = {0.0f, 0.0f, 0.0f};
    const float up[3]     = {0.0f, 1.0f, 0.0f};
    lookAt(eye, center, up, view);

    // Multiply projection and view to get the combined MVP matrix.
    mat4_mult(proj, view, mvp);
}

// -------------------------------
// Shader sources
// -------------------------------
// Vertex shader: increased point size for visibility.
const char* vertexShaderSource = R"(
    attribute vec3 a_position;
    uniform mat4 u_mvp;
    void main() {
        gl_Position = u_mvp * vec4(a_position, 1.0);
        gl_PointSize = 10.0;
    }
)";

const char* fragmentShaderSource = R"(
    precision mediump float;
    void main() {
        gl_FragColor = vec4(0.0, 0.5, 1.0, 1.0);
    }
)";

// -------------------------------
// Shader compilation utilities
// -------------------------------
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    if (!shader) {
        printf("Error creating shader\n");
        return 0;
    }
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint logLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 1) {
            char* log = new char[logLen];
            glGetShaderInfoLog(shader, logLen, nullptr, log);
            printf("Shader compile log:\n%s\n", log);
            delete[] log;
        }
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint createProgram(const char* vsSource, const char* fsSource) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vsSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fsSource);
    GLuint prog = glCreateProgram();
    if (!prog) {
        return 0;
    }
    glAttachShader(prog, vertexShader);
    glAttachShader(prog, fragmentShader);
    glBindAttribLocation(prog, 0, "a_position");
    glLinkProgram(prog);
    GLint linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint logLen = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 1) {
            char* log = new char[logLen];
            glGetProgramInfoLog(prog, logLen, nullptr, log);
            printf("Program link log:\n%s\n", log);
            delete[] log;
        }
        glDeleteProgram(prog);
        return 0;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return prog;
}

// -------------------------------
// Particle Initialization
// -------------------------------
void initParticles() {
    particles.resize(NUM_PARTICLES);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distPos(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distVel(-0.001f, 0.001f);
    
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        particles[i].pos[0] = distPos(gen);
        particles[i].pos[1] = distPos(gen);
        particles[i].pos[2] = distPos(gen);
        particles[i].vel[0] = distVel(gen);
        particles[i].vel[1] = distVel(gen);
        particles[i].vel[2] = distVel(gen);
    }
}

// -------------------------------
// Spatial Grid Collision Resolution
// -------------------------------
void resolveCollisions() {
    const int gridDim = static_cast<int>(2.0f / cellSize);  // e.g., 20 cells per axis
    const int totalCells = gridDim * gridDim * gridDim;
    std::vector< std::vector<int> > grid(totalCells);

    auto computeCellIndex = [=](const float pos[3]) -> int {
        int ix = static_cast<int>(floor((pos[0] + 1.0f) / cellSize));
        int iy = static_cast<int>(floor((pos[1] + 1.0f) / cellSize));
        int iz = static_cast<int>(floor((pos[2] + 1.0f) / cellSize));
        if (ix < 0) ix = 0; if (ix >= gridDim) ix = gridDim - 1;
        if (iy < 0) iy = 0; if (iy >= gridDim) iy = gridDim - 1;
        if (iz < 0) iz = 0; if (iz >= gridDim) iz = gridDim - 1;
        return ix + iy * gridDim + iz * gridDim * gridDim;
    };

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        int cellIndex = computeCellIndex(particles[i].pos);
        grid[cellIndex].push_back(i);
    }

    std::vector<float> repulsion(3 * NUM_PARTICLES, 0.0f);

    for (int x = 0; x < gridDim; ++x) {
        for (int y = 0; y < gridDim; ++y) {
            for (int z = 0; z < gridDim; ++z) {
                int cellIndex = x + y * gridDim + z * gridDim * gridDim;
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= gridDim) continue;
                    for (int dy = -1; dy <= 1; ++dy) {
                        int ny = y + dy;
                        if (ny < 0 || ny >= gridDim) continue;
                        for (int dz = -1; dz <= 1; ++dz) {
                            int nz = z + dz;
                            if (nz < 0 || nz >= gridDim) continue;
                            int neighborIndex = nx + ny * gridDim + nz * gridDim * gridDim;
                            for (int i : grid[cellIndex]) {
                                for (int j : grid[neighborIndex]) {
                                    if (i < j) {
                                        float dx = particles[i].pos[0] - particles[j].pos[0];
                                        float dy = particles[i].pos[1] - particles[j].pos[1];
                                        float dz = particles[i].pos[2] - particles[j].pos[2];
                                        float dist = sqrtf(dx*dx + dy*dy + dz*dz) + 1e-6f;
                                        if (dist < minDistance) {
                                            float force_mag = stiffness * powf((minDistance - dist), 2);
                                            float fx = force_mag * dx / dist;
                                            float fy = force_mag * dy / dist;
                                            float fz = force_mag * dz / dist;
                                            repulsion[3*i + 0] += fx;
                                            repulsion[3*i + 1] += fy;
                                            repulsion[3*i + 2] += fz;
                                            repulsion[3*j + 0] -= fx;
                                            repulsion[3*j + 1] -= fy;
                                            repulsion[3*j + 2] -= fz;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        float acc[3];
        acc[0] = repulsion[3*i + 0];
        acc[1] = repulsion[3*i + 1] + gravity[1];
        acc[2] = repulsion[3*i + 2];

        particles[i].vel[0] = (particles[i].vel[0] + acc[0] * dt) * damping;
        particles[i].vel[1] = (particles[i].vel[1] + acc[1] * dt) * damping;
        particles[i].vel[2] = (particles[i].vel[2] + acc[2] * dt) * damping;
        
        particles[i].pos[0] += particles[i].vel[0] * dt;
        particles[i].pos[1] += particles[i].vel[1] * dt;
        particles[i].pos[2] += particles[i].vel[2] * dt;

        for (int j = 0; j < 3; ++j) {
            if (particles[i].pos[j] < -1.0f) {
                particles[i].pos[j] = -1.0f;
                particles[i].vel[j] = -0.5f * particles[i].vel[j];
            }
            if (particles[i].pos[j] > 1.0f) {
                particles[i].pos[j] = 1.0f;
                particles[i].vel[j] = -0.5f * particles[i].vel[j];
            }
        }
    }
}

// -------------------------------
// Simulation Update
// -------------------------------
void updateSimulation() {
    resolveCollisions();
}

// -------------------------------
// OpenGL Rendering of Particles
// -------------------------------
void renderFrame() {
    // Clear the screen and depth buffer.
    glClearColor(0.3f, 0.0f, 0.0f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // --- Debug Test Block ---
    // Uncomment the following block to draw a single white test point at (0,0,0)
    /*
    std::vector<float> testPoint = {0.0f, 0.0f, 0.0f};
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, testPoint.size() * sizeof(float), testPoint.data());
    glUseProgram(program);
    // Set an identity MVP for the test.
    float identity[16] = {
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
    };
    glUniformMatrix4fv(uniform_mvp, 1, GL_FALSE, identity);
    glEnableVertexAttribArray(attrib_position);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(attrib_position, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_POINTS, 0, 1);
    glDisableVertexAttribArray(attrib_position);
    // End Test Block
    */

    // Update the VBO with the particle positions.
    std::vector<float> posData;
    posData.reserve(NUM_PARTICLES * 3);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        posData.push_back(particles[i].pos[0]);
        posData.push_back(particles[i].pos[1]);
        posData.push_back(particles[i].pos[2]);
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, posData.size() * sizeof(float), posData.data());

    glUseProgram(program);
    glUniformMatrix4fv(uniform_mvp, 1, GL_FALSE, mvp);

    glEnableVertexAttribArray(attrib_position);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(attrib_position, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
    glDisableVertexAttribArray(attrib_position);
}

// -------------------------------
// Main loop called by Emscripten
// -------------------------------
void main_loop() {
    frameCount++;
    updateSimulation();
    renderFrame();
    SDL_GL_SwapWindow(window);

    // Debug output every 60 frames (~once per second)
    if (frameCount % 60 == 0) {
        printf("Frame %d: Particle 0 position = (%f, %f, %f)\n",
               frameCount,
               particles[0].pos[0],
               particles[0].pos[1],
               particles[0].pos[2]);
    }
}

// -------------------------------
// Main: Initialization and Setup
// -------------------------------
int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize: %s\n", SDL_GetError());
        return 1;
    }
    
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

    window = SDL_CreateWindow("Particle Simulation (WebAssembly)",
                              SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              800, 600,
                              SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!window) {
        printf("Window could not be created: %s\n", SDL_GetError());
        return 1;
    }
    
    glContext = SDL_GL_CreateContext(window);
    if (!glContext) {
        printf("OpenGL context could not be created: %s\n", SDL_GetError());
        return 1;
    }
    
    // Set the viewport.
    glViewport(0, 0, 800, 600);

    // Enable depth testing for proper 3D rendering.
    glEnable(GL_DEPTH_TEST);

    program = createProgram(vertexShaderSource, fragmentShaderSource);
    if (!program) {
        printf("Failed to create shader program.\n");
        return 1;
    }
    attrib_position = 0; // Bound to "a_position"
    uniform_mvp = glGetUniformLocation(program, "u_mvp");

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // Compute the MVP using the updated camera parameters.
    computeMVP();
    initParticles();

    printf("Starting main loop...\n");
    emscripten_set_main_loop(main_loop, 0, 1);

    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
