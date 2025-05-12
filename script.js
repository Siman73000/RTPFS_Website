document.getElementById('setParticles').addEventListener('click', ()=>{
    const v = parseInt(prompt('Enter particle count (1–100000):', particleCount), 10);
    if (!isNaN(v) && v>=1 && v<=100000) { particleCount = v; resetParticles(); }
    document.getElementById('menu').checked = false;
  });

  // Fluid color picker
  document.getElementById('pickFluidColor').addEventListener('click', ()=>{
    const cp = document.getElementById('fluidColorPicker');
    cp.value = particleSystem.material.color.getStyle();
    cp.click();
    cp.oninput = () => {
      particleSystem.material.color.set(cp.value);
    };
    document.getElementById('menu').checked = false;
  });

  // Blur
  document.getElementById('setBlur').addEventListener('click', ()=>{
    const b = parseFloat(prompt('Enter blur amount (0–5):', (hBlur.uniforms.h.value*window.innerWidth/2).toFixed(2)));
    if (!isNaN(b) && b>=0 && b<=5) {
      hBlur.uniforms.h.value = b / window.innerWidth * 2;
      vBlur.uniforms.v.value = b / window.innerHeight * 2;
      renderer.domElement.style.filter = `blur(${b}px)`;
    }
    document.getElementById('menu').checked = false;
  });

  // Background color picker
  document.getElementById('pickBgColor').addEventListener('click', ()=>{
    const bgp = document.getElementById('bgColorPicker');
    bgp.value = renderer.getClearColor().getStyle();
    bgp.click();
    bgp.oninput = () => {
      renderer.setClearColor(bgp.value);
    };
    document.getElementById('menu').checked = false;
  });
