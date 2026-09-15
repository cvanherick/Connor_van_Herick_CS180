(function () {
  const root = document.documentElement;
  const themeButton = document.querySelector(".theme-toggle");
  const savedTheme = localStorage.getItem("project1-theme");

  if (savedTheme) root.dataset.theme = savedTheme;

  function currentTheme() {
    if (root.dataset.theme) return root.dataset.theme;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function updateThemeButton() {
    if (!themeButton) return;
    const theme = currentTheme();
    themeButton.textContent = theme === "dark" ? "Light" : "Dark";
    themeButton.setAttribute("aria-pressed", String(theme === "dark"));
  }

  themeButton?.addEventListener("click", () => {
    const nextTheme = currentTheme() === "dark" ? "light" : "dark";
    root.dataset.theme = nextTheme;
    localStorage.setItem("project1-theme", nextTheme);
    updateThemeButton();
  });
  updateThemeButton();

  const sectionLinks = Array.from(document.querySelectorAll('.project1-page .nav a[href^="#"]'));
  const sections = sectionLinks.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        sectionLinks.forEach((link) => link.setAttribute("aria-current", String(link.getAttribute("href") === `#${visible.target.id}`)));
      },
      { rootMargin: "-24% 0px -62% 0px", threshold: [0.1, 0.25, 0.5] }
    );
    sections.forEach((section) => observer.observe(section));
  }

  const comparisonSliders = Array.from(document.querySelectorAll(".comparison-slider"));
  comparisonSliders.forEach((slider) => {
    const range = slider.querySelector(".comparison-range");
    const before = slider.querySelector(".comparison-before");
    const handle = slider.querySelector(".comparison-handle");
    const updateComparison = () => {
      const position = `${range.value}%`;
      before.style.width = position;
      handle.style.left = position;
    };
    range.addEventListener("input", updateComparison);
    updateComparison();
  });

  const canvas = document.querySelector(".alignment-canvas");
  const alignmentStatus = document.querySelector(".alignment-demo-status");
  const resetAlignment = document.querySelector(".reset-alignment");
  const alignButton = document.querySelector(".align-button");
  const startingOffsets = {
    green: { dy: -12, dx: 7 },
    red: { dy: 15, dx: -9 },
  };

  if (canvas && alignmentStatus) {
    const context = canvas.getContext("2d");
    const demoImage = new Image();
    const offsets = { green: { ...startingOffsets.green }, red: { ...startingOffsets.red } };
    let sourcePixels;

    function drawAlignment() {
      if (!sourcePixels) return;
      const { width, height, data: source } = sourcePixels;
      const output = context.createImageData(width, height);
      const { data } = output;

      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const target = (y * width + x) * 4;
          const greenX = x - offsets.green.dx;
          const greenY = y - offsets.green.dy;
          const redX = x - offsets.red.dx;
          const redY = y - offsets.red.dy;
          data[target] = redX >= 0 && redX < width && redY >= 0 && redY < height ? source[(redY * width + redX) * 4] : 0;
          data[target + 1] = greenX >= 0 && greenX < width && greenY >= 0 && greenY < height ? source[(greenY * width + greenX) * 4 + 1] : 0;
          data[target + 2] = source[target + 2];
          data[target + 3] = 255;
        }
      }

      context.putImageData(output, 0, 0);
      alignmentStatus.textContent = `Green: (${offsets.green.dy}, ${offsets.green.dx}) · Red: (${offsets.red.dy}, ${offsets.red.dx})`;
    }

    demoImage.addEventListener("load", () => {
      const scale = Math.min(1, 900 / demoImage.naturalWidth);
      canvas.width = Math.round(demoImage.naturalWidth * scale);
      canvas.height = Math.round(demoImage.naturalHeight * scale);
      context.drawImage(demoImage, 0, 0, canvas.width, canvas.height);
      sourcePixels = context.getImageData(0, 0, canvas.width, canvas.height);
      drawAlignment();
    });
    demoImage.src = "CS180_fa2026_merged_photos/emir_out.jpg";

    document.querySelectorAll(".channel-control").forEach((control) => {
      const channel = control.dataset.channel;
      control.querySelectorAll("button").forEach((button) => {
        button.addEventListener("click", () => {
          offsets[channel].dy += Number(button.dataset.dy);
          offsets[channel].dx += Number(button.dataset.dx);
          drawAlignment();
        });
      });
    });

    resetAlignment?.addEventListener("click", () => {
      Object.assign(offsets.green, startingOffsets.green);
      Object.assign(offsets.red, startingOffsets.red);
      drawAlignment();
    });
    alignButton?.addEventListener("click", () => {
      offsets.green.dy = 0;
      offsets.green.dx = 0;
      offsets.red.dy = 0;
      offsets.red.dx = 0;
      drawAlignment();
    });
  }

  function metadataForImage(image) {
    const card = image.closest(".final-card");
    const caption = card?.querySelector("figcaption");
    const spans = caption ? Array.from(caption.querySelectorAll("span")) : [];
    const sideLabel = image.closest(".result-pair > div")?.querySelector("span")?.textContent?.trim();
    const title = caption?.querySelector("strong")?.textContent?.trim() || image.alt;
    return {
      after: image.currentSrc || image.src,
      alt: image.alt,
      method: [sideLabel, spans[0]?.textContent?.trim()].filter(Boolean).join(" · "),
      offsets: spans[1]?.textContent?.trim() || "",
      title,
    };
  }

  const galleryItems = Array.from(document.querySelectorAll(".final-card img, .comparison-slider")).map((element) => {
    if (element.classList.contains("comparison-slider")) {
      return {
        after: element.dataset.after,
        alt: element.querySelector(".comparison-after")?.alt || element.dataset.title,
        before: element.dataset.before,
        beforeAlt: element.querySelector(".comparison-before img")?.alt || "Raw NCC result",
        element,
        method: element.dataset.method || "",
        offsets: element.dataset.offsets || "",
        title: element.dataset.title || "Comparison",
      };
    }
    return { ...metadataForImage(element), element };
  });

  const lightbox = document.querySelector(".lightbox");
  const lightboxImage = lightbox?.querySelector("img");
  const lightboxTitle = lightbox?.querySelector("h2");
  const lightboxMethod = lightbox?.querySelector(".lightbox-method");
  const lightboxOffsets = lightbox?.querySelector(".lightbox-offsets");
  const lightboxCount = lightbox?.querySelector(".lightbox-count");
  const closeButton = lightbox?.querySelector(".lightbox-close");
  const previousButton = lightbox?.querySelector(".lightbox-previous");
  const nextButton = lightbox?.querySelector(".lightbox-next");
  const toggleButton = lightbox?.querySelector(".lightbox-toggle");
  let activeIndex = 0;
  let showBefore = false;
  let lastFocusedElement;

  function renderLightbox() {
    const item = galleryItems[activeIndex];
    if (!item || !lightboxImage || !lightboxTitle || !lightboxMethod || !lightboxOffsets || !lightboxCount || !toggleButton) return;
    const isBefore = showBefore && Boolean(item.before);
    lightboxImage.src = isBefore ? item.before : item.after;
    lightboxImage.alt = isBefore ? item.beforeAlt : item.alt;
    lightboxTitle.textContent = item.title;
    lightboxMethod.textContent = item.method;
    lightboxOffsets.textContent = item.offsets;
    lightboxOffsets.hidden = !item.offsets;
    lightboxCount.textContent = `${activeIndex + 1} / ${galleryItems.length}`;
    toggleButton.hidden = !item.before;
    toggleButton.textContent = isBefore ? "Show Final Result" : "Show Raw NCC";
  }

  function openLightbox(index, trigger) {
    if (!lightbox || !galleryItems[index]) return;
    activeIndex = index;
    showBefore = false;
    lastFocusedElement = trigger || document.activeElement;
    renderLightbox();
    lightbox.hidden = false;
    document.body.classList.add("lightbox-open");
    closeButton?.focus();
  }

  function closeLightbox() {
    if (!lightbox) return;
    lightbox.hidden = true;
    document.body.classList.remove("lightbox-open");
    lastFocusedElement?.focus?.();
  }

  function stepLightbox(direction) {
    activeIndex = (activeIndex + direction + galleryItems.length) % galleryItems.length;
    showBefore = false;
    renderLightbox();
  }

  galleryItems.forEach((item, index) => {
    if (item.element instanceof HTMLImageElement) {
      item.element.setAttribute("role", "button");
      item.element.tabIndex = 0;
      item.element.setAttribute("aria-label", `Open ${item.title} image`);
      item.element.addEventListener("click", () => openLightbox(index, item.element));
      item.element.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openLightbox(index, item.element);
        }
      });
      return;
    }

    const range = item.element.querySelector(".comparison-range");
    item.element.addEventListener("click", (event) => {
      if (event.target !== range) openLightbox(index, item.element);
    });
    range?.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        openLightbox(index, range);
      }
    });
  });

  closeButton?.addEventListener("click", closeLightbox);
  previousButton?.addEventListener("click", () => stepLightbox(-1));
  nextButton?.addEventListener("click", () => stepLightbox(1));
  toggleButton?.addEventListener("click", () => {
    showBefore = !showBefore;
    renderLightbox();
  });
  lightbox?.addEventListener("click", (event) => {
    if (event.target === lightbox) closeLightbox();
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !lightbox?.hidden) closeLightbox();
    if (lightbox?.hidden) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      stepLightbox(-1);
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      stepLightbox(1);
    }
  });

  const backToTop = document.querySelector(".back-to-top");
  backToTop?.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
  window.addEventListener("scroll", () => backToTop?.classList.toggle("is-visible", window.scrollY > 700), { passive: true });
})();
