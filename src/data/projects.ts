import type { Project } from "./types";

export const projects = [
  {
    slug: "rag-faithfulness-probe",
    title: "RAG Faithfulness Probe",
    summary:
      "A local, reproducible evaluation harness that decomposes RAG answers into atomic claims and compares NLI and LLM-judge assessments of source support.",
    category: "ai-reliability",
    technologies: [
      "Python",
      "LLaMA 3.1",
      "DeBERTa",
      "Ollama",
      "Docker",
      "RAGTruth",
    ],
    links: [
      {
        label: "GitHub",
        url: "https://github.com/SOUMEE2000/rag-faithfulness-probe",
      },
    ],
    sources: ["public-repository"],
    featuredRank: 1,
  },
  {
    slug: "developer-interaction-models",
    title: "Developer Interaction Models",
    summary:
      "Agent-based simulations of collaboration networks in large software ecosystems, validated against Eclipse, Android, and OpenStack data.",
    period: {
      start: "2023-02",
      end: "2023-08",
      label: "Feb 2023 - Aug 2023",
    },
    category: "complex-systems",
    technologies: ["NetLogo", "Java", "Streamlit", "SQL", "Python"],
    links: [
      {
        label: "GitHub",
        url: "https://github.com/SOUMEE2000/Netlogo-Model-of-Developer-Interaction",
      },
    ],
    sources: ["resume", "portfolio", "public-repository"],
    featuredRank: 2,
    image: "/assets/projects/developer-interaction.png",
    imageAlt:
      "NetLogo interface showing a simulated software developer interaction network",
  },
  {
    slug: "ragkit",
    title: "RAGKit",
    summary:
      "A production retrieval-augmented document analysis system that produces structured, source-grounded summaries over heterogeneous document collections.",
    period: {
      start: "2023-12",
      end: "present",
      label: "Dec 2023 - Present",
    },
    category: "information-systems",
    technologies: ["Python", "RAG", "Elasticsearch", "LLMs", "REST APIs"],
    links: [],
    sources: ["resume"],
    featuredRank: 3,
  },
  {
    slug: "route-map-composer",
    title: "Route Map Composer",
    summary:
      "A configurable geospatial visualization tool for turning structured route and region data into interactive maps and exportable visual reports.",
    period: {
      start: "2023-12",
      end: "present",
      label: "Dec 2023 - Present",
    },
    category: "information-systems",
    technologies: ["Python", "Leaflet", "Folium", "Geopy", "Docker"],
    links: [],
    sources: ["resume"],
    featuredRank: 4,
  },
  {
    slug: "applicant-tracking-system",
    title: "Applicant Tracking System",
    summary:
      "A Streamlit application that compares resumes with job descriptions using contextual embeddings, cosine similarity, and skill classification.",
    period: {
      start: "2023-03",
      end: "2023-03",
      label: "Mar 2023",
    },
    category: "machine-learning",
    technologies: ["Python", "BERT", "NLP", "Cosine Similarity", "Streamlit"],
    links: [
      {
        label: "Live demo",
        url: "https://soumee2000-applicant-tracking-system-application-tqrpm0.streamlit.app/",
      },
      {
        label: "GitHub",
        url: "https://github.com/SOUMEE2000/Applicant_Tracking_System",
      },
    ],
    sources: ["resume", "portfolio", "public-repository"],
    featuredRank: 5,
    image: "/assets/projects/applicant-tracking-system.png",
    imageAlt:
      "Applicant Tracking System results interface showing a resume match score",
  },
  {
    slug: "machine-learning-stash",
    title: "Machine Learning Stash",
    summary:
      "A research notebook collection spanning U-Nets, retinal and dendritic-spine segmentation, neural networks, sentiment analysis, transfer learning, and natural language generation.",
    period: {
      start: "2020-11",
      end: "2023-02",
      label: "Nov 2020 - Feb 2023",
    },
    category: "machine-learning",
    technologies: [
      "TensorFlow",
      "scikit-learn",
      "scikit-image",
      "C++",
      "U-Net",
    ],
    links: [
      {
        label: "GitHub",
        url: "https://github.com/SOUMEE2000/Machine-Learning-Stash",
      },
    ],
    sources: ["resume", "portfolio", "public-repository"],
    featuredRank: 6,
  },
  {
    slug: "multilingual-public-source-aggregation",
    title: "Multilingual Public-Source Aggregation",
    summary:
      "A multilingual pipeline that collects and normalizes reporting and public information from regional web and feed sources for analysis.",
    period: {
      start: "2023-12",
      end: "present",
      label: "Dec 2023 - Present",
    },
    category: "information-systems",
    technologies: ["Python", "RSS", "Playwright", "Data Processing"],
    links: [],
    sources: ["resume"],
  },
  {
    slug: "ct-perfusion-stroke-analysis",
    title: "CT Perfusion Stroke Analysis",
    summary:
      "Research pipelines for preparing CT perfusion data, denoising medical images, and estimating blood-flow parameters used to identify stroke regions.",
    period: {
      start: "2022-07",
      end: "2022-10",
      label: "Jul 2022 - Oct 2022",
    },
    category: "medical-imaging",
    technologies: [
      "Python",
      "Conditional GANs",
      "DICOM",
      "NIfTI",
      "Time-series Analysis",
    ],
    links: [],
    sources: ["resume"],
  },
  {
    slug: "dendritic-spine-segmentation",
    title: "Dendritic Spine Image Segmentation",
    summary:
      "A meta-consensus binarization and U-Net workflow for segmenting dendritic spine images, with graph-based measurements integrated into JYNIA.",
    period: {
      start: "2021-07",
      end: "2023-01",
      label: "Jul 2021 - Jan 2023",
    },
    category: "medical-imaging",
    technologies: ["Python", "C++", "U-Net", "Transfer Learning", "Linux"],
    links: [
      {
        label: "JYNIA",
        url: "https://jynia.org/",
      },
      {
        label: "Publication",
        url: "https://doi.org/10.1007/978-981-99-1509-5_25",
      },
    ],
    sources: ["resume"],
  },
  {
    slug: "diabetic-retinopathy-detection",
    title: "Diabetic Retinopathy Detection",
    summary:
      "Retinal vessel segmentation experiments using clustering, support-vector machines, neural networks, and class-balancing methods.",
    period: {
      start: "2021-04",
      end: "2021-07",
      label: "Apr 2021 - Jul 2021",
    },
    category: "medical-imaging",
    technologies: [
      "Python",
      "Image Segmentation",
      "K-Means",
      "SVM",
      "SMOTE",
      "ADASYN",
    ],
    links: [],
    sources: ["resume"],
  },
  {
    slug: "dress-sizing",
    title: "Dress Sizing",
    summary:
      "An image-processing prototype that adapts clothing imagery between body structures using key-point detection and seam-carving techniques.",
    period: {
      start: "2022-08",
      end: "2022-08",
      label: "Aug 2022",
    },
    category: "machine-learning",
    technologies: [
      "Image Processing",
      "Key-point Detection",
      "Seam Carving",
      "Command-line Tools",
    ],
    links: [
      {
        label: "GitHub",
        url: "https://github.com/SOUMEE2000/Dress-Sizing",
      },
    ],
    sources: ["portfolio", "public-repository"],
  },
  {
    slug: "reading-project",
    title: "Machine Learning Reading Project",
    summary:
      "Theory-to-code studies of neural networks, continuous bag-of-words models, graph neural networks, and sentiment analysis.",
    category: "machine-learning",
    technologies: [
      "Python",
      "Neural Networks",
      "CBOW",
      "Graph Neural Networks",
    ],
    links: [
      {
        label: "Sentiment analysis notebooks",
        url: "https://github.com/SOUMEE2000/Machine-Learning-Stash/tree/main/4.%20Sentiment%20Analysis",
      },
    ],
    sources: ["resume", "public-repository"],
  },
] satisfies readonly Project[];
