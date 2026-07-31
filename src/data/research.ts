import type { ResearchItem } from "./types";

export const researchItems = [
  {
    institution: "Heritage Institute of Technology",
    role: "Bachelor's Thesis Researcher",
    project:
      "Emergent Structure in Large-Scale Software Ecosystems via Agent-Based Modelling",
    location: "Kolkata, India",
    period: {
      start: "2023-02",
      end: "2023-08",
      label: "Feb 2023 - Aug 2023",
    },
    summary:
      "Studied how local developer interactions produce global collaboration structures in large software ecosystems.",
    highlights: [
      "Built agent-based simulations of evolving developer collaboration networks.",
      "Validated synthetic network behaviour against Eclipse, Android, and OpenStack ecosystems.",
      "Produced two manuscripts and supporting analysis tools for large CSV corpora.",
    ],
    technologies: ["NetLogo", "Java", "Streamlit", "SQL", "Complex Networks"],
    links: [
      {
        label: "GitHub",
        url: "https://github.com/SOUMEE2000/Netlogo-Model-of-Developer-Interaction",
      },
      {
        label: "Preprint",
        url: "https://doi.org/10.2139/ssrn.5734055",
      },
    ],
  },
  {
    institution: "University of Calgary",
    role: "Deep Learning Research Intern, MITACS Fellow",
    project: "CT Perfusion Stroke Analysis using Machine Learning",
    location: "Calgary, Canada",
    period: {
      start: "2022-07",
      end: "2022-10",
      label: "Jul 2022 - Oct 2022",
    },
    summary:
      "Investigated machine-learning workflows for preparing and analysing CT perfusion stroke data.",
    highlights: [
      "Prepared data pipelines for DICOM and NIfTI patient imaging data.",
      "Applied conditional GAN and Fourier-based denoising techniques.",
      "Estimated cerebral blood-flow parameters used to delineate stroke regions.",
    ],
    technologies: [
      "Python",
      "Shell",
      "Linux",
      "DICOM",
      "NIfTI",
      "Conditional GANs",
    ],
    links: [],
  },
  {
    institution: "Jadavpur University",
    role: "Application Developer",
    project: "Denoising Medical Images via Binarization and Deep Learning",
    location: "Kolkata, India",
    period: {
      start: "2021-07",
      end: "2023-01",
      label: "Jul 2021 - Jan 2023",
    },
    summary:
      "Developed image-segmentation and analysis methods for dendritic spine microscopy.",
    highlights: [
      "Combined local and global thresholding with U-Net segmentation in a meta-consensus model.",
      "Applied graph algorithms to spine-density and dendritic-loop measurements.",
      "Contributed methods for integration into the JYNIA neuroscience software.",
    ],
    technologies: ["C++", "Python", "U-Net", "Transfer Learning", "Linux"],
    links: [
      {
        label: "JYNIA",
        url: "https://jynia.org/",
      },
      {
        label: "Springer chapter",
        url: "https://doi.org/10.1007/978-981-99-1509-5_25",
      },
    ],
  },
  {
    institution: "Heritage Institute of Technology",
    role: "Undergraduate Researcher",
    project: "Diabetic Retinopathy Detection via Supervised Learning",
    location: "Kolkata, India",
    period: {
      start: "2021-04",
      end: "2021-07",
      label: "Apr 2021 - Jul 2021",
    },
    summary:
      "Compared classical and neural approaches to retinal blood-vessel segmentation.",
    highlights: [
      "Prepared pixel-level vessel and background data from retinal fundus images.",
      "Compared K-Means, support-vector machines, and neural-network models.",
      "Applied SMOTE and ADASYN to address class imbalance.",
    ],
    technologies: [
      "Python",
      "Image Segmentation",
      "K-Means",
      "SVM",
      "SMOTE",
      "ADASYN",
    ],
    links: [],
  },
] satisfies readonly ResearchItem[];
