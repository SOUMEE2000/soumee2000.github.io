import type { Education, Fellowship } from "./types";

export const education = [
  {
    institution: "Heritage Institute of Technology",
    program: "B.Tech in Computer Science and Engineering",
    location: "Kolkata, India",
    period: {
      start: "2019-08",
      end: "2023-08",
      label: "Aug 2019 - Aug 2023",
    },
    grade: "GPA: 9.12/10",
    coursework: [
      "Probability and Statistics",
      "Data Science",
      "Natural Language Processing",
      "Pattern Recognition",
      "Numerical Methods",
      "Relational Databases",
      "NoSQL Databases",
    ],
    kind: "degree",
  },
  {
    institution: "Indian Statistical Institute",
    program: "Winter Workshop on Advanced Machine Learning",
    location: "Kolkata, India",
    period: {
      start: "2023-01",
      end: "2023-03",
      label: "Jan 2023 - Mar 2023",
    },
    kind: "professional-development",
  },
] satisfies readonly Education[];

export const fellowship = {
  name: "MITACS Globalink Research Internship",
  institution: "University of Calgary",
  location: "Calgary, Canada",
  period: {
    start: "2022-07",
    end: "2022-10",
    label: "Jul 2022 - Oct 2022",
  },
  selection:
    "Selected for a funded research internship from a global pool of 20,000 applicants.",
  summary:
    "Conducted deep-learning research on CT perfusion stroke analysis in an international research environment.",
} satisfies Fellowship;
