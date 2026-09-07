#pragma once
#include <glm/glm.hpp>
#include <string>

namespace Boids {
    void initSimulation(int N);
    void stepSimulationNaive(float dt);
    void stepSimulationScatteredGrid(float dt);
    void stepSimulationCoherentGrid(float dt);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

    void endSimulation();
    void unitTest();
    void specUnitTest(std::string name, glm::vec3* before_pos, glm::vec3* before_vel1,
        glm::vec3* before_vel2, glm::vec3* exp_pos, glm::vec3* exp_vel1,
        glm::vec3* exp_vel2, int num_boids, float dt);
    void specUnitTest(std::string name, glm::vec3* before_pos, glm::vec3* before_vel1,
        glm::vec3* before_vel2, glm::vec3* exp_pos, glm::vec3* exp_vel1,
        glm::vec3* exp_vel2, int num_boids, float dt, int mode);

    enum Mode {
        NAIVE,
        SCATTERED,
        COHERENT
    };

}
