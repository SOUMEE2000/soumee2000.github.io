export interface DateRange {
  start: string;
  end: string;
  label: string;
}

export interface ExternalLink {
  label: string;
  url: string;
}

export interface Profile {
  name: string;
  roles: readonly string[];
  headline: string;
  summary: string;
  location: string;
  email: string;
  siteUrl: string;
  socialLinks: readonly ExternalLink[];
}

export type ProjectCategory =
  | "ai-reliability"
  | "information-systems"
  | "complex-systems"
  | "medical-imaging"
  | "machine-learning"
  | "developer-tools";

export type ProjectSource = "resume" | "portfolio" | "public-repository";

export interface Project {
  slug: string;
  title: string;
  summary: string;
  period?: DateRange;
  category: ProjectCategory;
  technologies: readonly string[];
  links: readonly ExternalLink[];
  sources: readonly ProjectSource[];
  featuredRank?: number;
  image?: string;
  imageAlt?: string;
}

export interface Experience {
  organization: string;
  role: string;
  location: string;
  period: DateRange;
  summary: string;
  highlights: readonly string[];
  skills: readonly string[];
}

export interface ResearchItem {
  institution: string;
  role: string;
  project: string;
  location: string;
  period: DateRange;
  summary: string;
  highlights: readonly string[];
  technologies: readonly string[];
  links: readonly ExternalLink[];
}

export type PublicationStatus = "published" | "accepted";

export interface Publication {
  title: string;
  authors: readonly string[];
  year: number;
  venue: string;
  type: "book-chapter" | "journal-article";
  status: PublicationStatus;
  pages?: string;
  doi: string;
  url: string;
}

export interface Education {
  institution: string;
  program: string;
  location: string;
  period: DateRange;
  grade?: string;
  coursework?: readonly string[];
  kind: "degree" | "professional-development";
}

export interface Fellowship {
  name: string;
  institution: string;
  location: string;
  period: DateRange;
  selection: string;
  summary: string;
}

export interface SkillGroup {
  name: string;
  skills: readonly string[];
}
