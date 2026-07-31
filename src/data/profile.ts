import type { Profile } from "./types";

export const profile = {
  name: "Soumee Mukherjee",
  roles: ["ML Researcher", "Software Developer"],
  headline:
    "Building reliable AI, information systems, and tools for understanding complex systems.",
  summary:
    "ML researcher and software developer working across retrieval-augmented generation, information trustworthiness, multi-agent behaviour, and computer vision. Her work combines production engineering with published research in deep learning and complex systems modelling.",
  location: "Delhi, India",
  email: "soumee.muk@gmail.com",
  siteUrl: "https://soumee2000.github.io/",
  socialLinks: [
    {
      label: "LinkedIn",
      url: "https://www.linkedin.com/in/soumee-mukherjee-6683721a1/",
    },
    {
      label: "GitHub",
      url: "https://github.com/SOUMEE2000",
    },
  ],
} satisfies Profile;
