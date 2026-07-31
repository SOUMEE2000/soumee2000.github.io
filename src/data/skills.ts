import type { SkillGroup } from "./types";

export const skillGroups = [
  {
    name: "Programming",
    skills: ["Python", "Java", "C", "C++", "Shell", "SQL", "LaTeX"],
  },
  {
    name: "AI, NLP, and RAG",
    skills: [
      "PyTorch",
      "TensorFlow",
      "Hugging Face",
      "scikit-learn",
      "LLMs",
      "Retrieval-Augmented Generation",
      "Encoder-Decoder Architectures",
      "Pandas",
    ],
  },
  {
    name: "Computer Vision",
    skills: ["OpenCV", "MediaPipe", "U-Net", "Image Segmentation"],
  },
  {
    name: "Data and Search",
    skills: ["Elasticsearch", "NumPy", "MySQL", "PostgreSQL", "MongoDB"],
  },
  {
    name: "Simulation and Geospatial",
    skills: ["NetLogo", "Folium", "Leaflet", "Geopy"],
  },
  {
    name: "Infrastructure and Tools",
    skills: ["Docker", "Playwright", "Linux", "SLURM", "Git", "REST APIs"],
  },
  {
    name: "Web",
    skills: ["HTML", "CSS", "JavaScript", "Streamlit"],
  },
] satisfies readonly SkillGroup[];
