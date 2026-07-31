import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://soumee2000.github.io/",
    title: "Soumee Mukherjee",
    description:
      "Machine learning researcher and software developer working across reliable AI systems, complex systems modelling, and applied research.",
    author: "Soumee Mukherjee",
    profile: "https://soumee2000.github.io/about/",
    ogImage: "default-og.jpg",
    lang: "en",
    timezone: "Asia/Kolkata",
    dir: "ltr",
  },
  posts: {
    perPage: 6,
    perIndex: 3,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: false,
    showArchives: false,
    showBackButton: true,
    editPost: {
      enabled: false,
    },
    search: "pagefind",
  },
  socials: [
    { name: "github", url: "https://github.com/SOUMEE2000" },
    {
      name: "linkedin",
      url: "https://www.linkedin.com/in/soumee-mukherjee-6683721a1/",
    },
    { name: "mail", url: "mailto:soumee.muk@gmail.com" },
  ],
  shareLinks: [
    { name: "whatsapp", url: "https://wa.me/?text=" },
    { name: "facebook", url: "https://www.facebook.com/sharer.php?u=" },
    { name: "x",        url: "https://x.com/intent/post?url=" },
    { name: "telegram", url: "https://t.me/share/url?url=" },
    { name: "pinterest", url: "https://pinterest.com/pin/create/button/?url=" },
    { name: "mail",     url: "mailto:?subject=See%20this%20post&body=" },
  ],
});
