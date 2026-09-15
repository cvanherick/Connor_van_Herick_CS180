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
})();
