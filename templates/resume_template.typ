// template.typ
// A clean, ATS-friendly single-column resume layout inspired by the widely-used
// "Jake's Resume" LaTeX template, rebuilt in Typst for fast/safe programmatic
// injection (Typst's syntax is far less prone to silent breakage than LaTeX's
// brace-heavy \resumeSubheading{}{}{}{} macros when a script is writing into it).
//
// This file is a TEMPLATE: resume_builder.py fills in the {{PLACEHOLDER}} tokens
// below with tailored content per job application, then runs
// `typst compile <filled_file>.typ <output>.pdf`.

#set page(margin: (x: 0.5in, y: 0.32in))
#set text(font: "New Computer Modern", size: 9.3pt)
#set par(justify: false, leading: 0.38em)
#set list(indent: 0em, spacing: 0.22em, marker: [•])

// Section heading style: small caps, bold, with a horizontal rule underneath
#let section_heading(title) = [
  #v(0.1em)
  #text(weight: "bold", size: 10pt, tracking: 0.5pt)[#upper(title)]
  #v(-0.5em)
  #line(length: 100%, stroke: 0.6pt)
  #v(0.02em)
]

// A single experience/project entry: title line + subtitle line + bullets
#let entry(title, subtitle, dates, location, bullets) = [
  #grid(
    columns: (1fr, auto),
    text(weight: "bold")[#title], text(weight: "bold")[#dates]
  )
  #grid(
    columns: (1fr, auto),
    text(style: "italic")[#subtitle], text(style: "italic")[#location]
  )
  #v(0.02em)
  #for b in bullets [
    - #b
  ]
  #v(0.08em)
]

// ===================== HEADER =====================
#align(center)[
  #text(size: 18pt, weight: "bold")[{{NAME}}]

  #v(-0.3em)
  {{CONTACT_LINE}}
]

#v(0.2em)

// ===================== EDUCATION =====================
#section_heading("Education")
{{EDUCATION_ENTRIES}}

// ===================== EXPERIENCE =====================
#section_heading("Experience")
{{EXPERIENCE_ENTRIES}}

// ===================== PROJECTS =====================
#section_heading("Projects")
{{PROJECT_ENTRIES}}

// ===================== SKILLS =====================
#section_heading("Skills")
{{SKILLS_BLOCK}}
