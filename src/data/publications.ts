import type { Publication } from "./types";

export const publications = [
  {
    title:
      "On Separation and Connection: A Multi-Systems Study of Developer Interaction using Agent-Based Models",
    authors: ["Soumee Mukherjee", "Subhajit Datta", "Subhashis Majumder"],
    year: 2025,
    venue: "Information and Software Technology",
    type: "journal-article",
    status: "accepted",
    doi: "10.2139/ssrn.5734055",
    url: "https://doi.org/10.2139/ssrn.5734055",
  },
  {
    title:
      "A Meta-consensus Strategy for Binarization of Dendritic Spines Images",
    authors: [
      "Shauvik Paul",
      "Nirmal Das",
      "Subhrabesh Dutta",
      "Dipannita Banerjee",
      "Soumee Mukherjee",
      "Subhadip Basu",
    ],
    year: 2023,
    venue:
      "Proceedings of International Conference on Data, Electronics and Computing, Algorithms for Intelligent Systems, Springer",
    type: "book-chapter",
    status: "published",
    pages: "269-278",
    doi: "10.1007/978-981-99-1509-5_25",
    url: "https://doi.org/10.1007/978-981-99-1509-5_25",
  },
] satisfies readonly Publication[];
