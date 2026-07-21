import katex from 'katex'
import 'katex/dist/katex.min.css'

export function LatexFormula({ formula }: { formula: string }) {
  const html = katex.renderToString(formula, {
    displayMode: true,
    throwOnError: false,
    strict: 'warn',
    trust: false,
    output: 'htmlAndMathml',
  })

  return (
    <div
      className="formula-block"
      data-latex={formula}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
