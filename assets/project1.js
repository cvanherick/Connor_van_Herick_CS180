(function () {
  const root = document.documentElement;
  const themeButton = document.querySelector(".theme-toggle");
  const savedTheme = localStorage.getItem("project1-theme");

  if (savedTheme) {
    root.dataset.theme = savedTheme;
  }

  function currentTheme() {
    if (root.dataset.theme) {
      return root.dataset.theme;
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function updateThemeButton() {
    if (!themeButton) {
      return;
    }
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
  const sections = sectionLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

        if (!visible) {
          return;
        }

        sectionLinks.forEach((link) => {
          link.setAttribute("aria-current", String(link.getAttribute("href") === `#${visible.target.id}`));
        });
      },
      { rootMargin: "-24% 0px -62% 0px", threshold: [0.1, 0.25, 0.5] }
    );

    sections.forEach((section) => observer.observe(section));
  }

  const lightbox = document.querySelector(".lightbox");
  const lightboxImage = lightbox?.querySelector("img");
  const lightboxCaption = lightbox?.querySelector("p");
  const closeButton = lightbox?.querySelector(".lightbox-close");
  const previewImages = document.querySelectorAll(".compare-row img, .source-plate-grid img");

  function closeLightbox() {
    if (!lightbox || !lightboxImage || !lightboxCaption) {
      return;
    }
    lightbox.hidden = true;
    lightboxImage.src = "";
    lightboxCaption.textContent = "";
  }

  previewImages.forEach((image) => {
    image.addEventListener("click", () => {
      if (!lightbox || !lightboxImage || !lightboxCaption) {
        return;
      }
      lightboxImage.src = image.currentSrc || image.src;
      lightboxImage.alt = image.alt;
      lightboxCaption.textContent =
        image.closest("figure")?.querySelector("figcaption")?.textContent || image.alt;
      lightbox.hidden = false;
    });
  });

  closeButton?.addEventListener("click", closeLightbox);
  lightbox?.addEventListener("click", (event) => {
    if (event.target === lightbox) {
      closeLightbox();
    }
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeLightbox();
    }
  });

  const backToTop = document.querySelector(".back-to-top");
  backToTop?.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));

  window.addEventListener(
    "scroll",
    () => {
      backToTop?.classList.toggle("is-visible", window.scrollY > 700);
    },
    { passive: true }
  );
})();
