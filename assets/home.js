(function () {
  document.querySelectorAll(".dolly-preview").forEach((preview) => {
    const motionImage = preview.querySelector(".preview-motion");
    const source = motionImage?.dataset.src;

    if (!motionImage || !source) return;

    function playDollyZoom() {
      motionImage.src = `${source}?play=${Date.now()}`;
    }

    function stopDollyZoom() {
      motionImage.removeAttribute("src");
    }

    const card = preview.closest(".project-card");
    card?.addEventListener("pointerenter", playDollyZoom);
    card?.addEventListener("pointerleave", stopDollyZoom);
    card?.addEventListener("focusin", playDollyZoom);
    card?.addEventListener("focusout", stopDollyZoom);
  });

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  document.querySelectorAll(".project3-preview").forEach((preview) => {
    const card = preview.closest(".project-card");

    function playDenoising() {
      if (reducedMotion.matches) return;
      preview.classList.remove("is-denoising");
      requestAnimationFrame(() => requestAnimationFrame(() => {
        if (!reducedMotion.matches) preview.classList.add("is-denoising");
      }));
    }

    preview.addEventListener("animationend", (event) => {
      if (event.animationName === "p3-noise-reveal") {
        preview.classList.remove("is-denoising");
      }
    });
    reducedMotion.addEventListener("change", () => {
      preview.classList.remove("is-denoising");
    });
    card?.addEventListener("pointerenter", playDenoising);
    card?.addEventListener("focusin", playDenoising);

    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          playDenoising();
          observer.disconnect();
        }
      }, { threshold: 0.25 });
      observer.observe(preview);
    }
  });
})();
