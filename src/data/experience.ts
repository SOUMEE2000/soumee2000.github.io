import type { Experience } from "./types";

export const experiences = [
  {
    organization: "Cabinet Secretariat, Government of India",
    role: "Software Developer",
    location: "Delhi, India",
    period: {
      start: "2023-12",
      end: "present",
      label: "Dec 2023 - Present",
    },
    summary:
      "Builds production information-retrieval, public-source analysis, and geospatial visualization capabilities.",
    highlights: [
      "Develops retrieval-augmented document analysis with source-grounded outputs.",
      "Builds multilingual public-source aggregation and analysis workflows.",
      "Creates configurable geospatial tools for interactive maps and visual reports.",
    ],
    skills: [
      "Python",
      "Information Retrieval",
      "RAG",
      "Data Pipelines",
      "Geospatial Visualization",
    ],
  },
  {
    organization: "Calcutta Electric Supply Corporation",
    role: "Internship Trainee",
    location: "Kolkata, India",
    period: {
      start: "2023-11",
      end: "2023-12",
      label: "Nov 2023 - Dec 2023",
    },
    summary:
      "Worked with the Security Operations Centre supporting critical network monitoring and process automation.",
    highlights: [
      "Supported Internet leased-line monitoring for metropolitan power distribution infrastructure.",
      "Automated operational processes on the Zabbix monitoring server.",
    ],
    skills: ["Network Security", "Zabbix", "Palo Alto Firewall", "PuTTY"],
  },
] satisfies readonly Experience[];
