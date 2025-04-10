document.addEventListener('DOMContentLoaded', function () {
    const hamburger = document.getElementById('hamburger');
    const navItems = document.getElementById('nav-items');

    hamburger.addEventListener('click', function () {
        hamburger.classList.toggle('active');
        navItems.classList.toggle('show');
    });
});


window.addEventListener('DOMContentLoaded', () => {
    const inputField = document.getElementById('input1');
    let particleCount = 10000;

    inputField.addEventListener('input', () => {
        const val = parseInt(inputField.value);
        if (!isNaN(val) && val > 0 && val <= 100000) {
            particleCount = val;
            resetParticles();
        }
    });

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.z = 3;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.querySelector('main').appendChild(renderer.domElement);

    let particleSystem, positions, velocities;

    const mockModel = {
        predict: async function(inputTensor) {
            return tf.randomUniform(inputTensor.shape, -0.002, 0.002);
        }
    };

    function resetParticles() {
        if (particleSystem) {
            scene.remove(particleSystem);
        }

        positions = new Float32Array(particleCount * 3);
        velocities = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {
            const i3 = i * 3;
            positions[i3 + 0] = (Math.random() - 0.5) * 2;
            positions[i3 + 1] = (Math.random() - 0.5) * 2;
            positions[i3 + 2] = (Math.random() - 0.5) * 2;

            velocities[i3 + 0] = (Math.random() - 0.5) * 0.01;
            velocities[i3 + 1] = (Math.random() - 0.5) * 0.01;
            velocities[i3 + 2] = (Math.random() - 0.5) * 0.01;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const material = new THREE.PointsMaterial({ color: 0x3399ff, size: 0.02 });
        particleSystem = new THREE.Points(geometry, material);

        scene.add(particleSystem);
    }

    resetParticles();

    async function animate() {
        requestAnimationFrame(animate);

        const posArray = new Float32Array(particleCount * 3);
        posArray.set(positions);

        const posTensor = tf.tensor2d(posArray, [particleCount, 3]);
        const deltaV = await mockModel.predict(posTensor);
        const deltaVArray = await deltaV.data();

        // Apply updates to velocity and position
        for (let i = 0; i < particleCount; i++) {
            const i3 = i * 3;

            velocities[i3 + 0] += deltaVArray[i3 + 0];
            velocities[i3 + 1] += deltaVArray[i3 + 1];
            velocities[i3 + 2] += deltaVArray[i3 + 2];

            positions[i3 + 0] += velocities[i3 + 0];
            positions[i3 + 1] += velocities[i3 + 1];
            positions[i3 + 2] += velocities[i3 + 2];

            for (let axis = 0; axis < 3; axis++) {
                if (positions[i3 + axis] > 1 || positions[i3 + axis] < -1) {
                    velocities[i3 + axis] *= -0.8;
                }
            }
        }

        tf.dispose([posTensor, deltaV]);

        particleSystem.geometry.attributes.position.needsUpdate = true;
        renderer.render(scene, camera);
    }

    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
});
