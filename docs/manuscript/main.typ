// From https://github.com/typst/typst/issues/2196#issuecomment-1728135476
#let to-string(content) = {
  if content.has("text") {
    content.text
  } else if content.has("children") {
    content.children.map(to-string).join("")
  } else if content.has("body") {
    to-string(content.body)
  } else if content == [ ] {
    " "
  }
}

#let title = [Current genomic deep learning architectures generalize across grass species but not alleles]

#set document(
  author: "Travis Wrightsman",
  title: to-string(title),
  date: datetime(year: 2024, month: 4, day: 11)
)
#set page(paper: "us-letter", numbering: "1", number-align: center)
#set text(font: "Linux Libertine", lang: "en")

// Title
#align(center)[
  #block(
    text(weight: 700, 1.75em)[#title]
  )
]

#v(2em, weak: true)

// Authors
#align(center)[
  Travis Wrightsman#super[1],
  Taylor H. Ferebee#super[2],
  M. Cinta Romay#super[3],
  Taylor AuBuchon-Elder#super[4],
  Alyssa R. Phillips#super[5],
  Michael Syring#super[6],
  Elizabeth A. Kellogg#super[4],
  Edward S. Buckler#super[1,3,7]
]

// Affiliations
#super[1]Section of Plant Breeding and Genetics, Cornell University, Ithaca, NY, USA 14853
#super[2]Department of Computational Biology, Cornell University, Ithaca, NY, USA 14853
#super[3]Institute for Genomic Diversity, Cornell University, Ithaca, NY, USA 14853
#super[4]Donald Danforth Plant Science Center, St. Louis, MO, USA 63132
#super[5]Department of Evolution and Ecology, University of California, Davis, CA, USA 95616
#super[6]Iowa State University, Ames, IA, USA 50011
#super[7]Agricultural Research Service, United States Department of Agriculture, Ithaca, NY, USA 14853

#set par(justify: true)

// Abstract
#align(center)[
  #heading(outlined: false, numbering: none, text(0.85em, [Abstract]))
]
#include "content/00-abstract.typ"
#v(2em, weak: true)

// Body
#include "content/01-introduction.typ"
#include "content/02-results.typ"
#include "content/03-discussion.typ"
#include "content/04-methods.typ"
#include "content/05-acknowledgements.typ"

// References
#include "content/90-references.typ"

// Supplemental
#include "content/91-supplemental.typ"
