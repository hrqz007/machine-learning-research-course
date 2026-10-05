// Convert trusted course TeX into self-contained SVGs without network access.
const fs = require('fs');
const path = require('path');
const root = path.dirname(require.resolve('mathjax-full/js/mathjax.js'));
const {mathjax} = require(root+'/mathjax.js');
const {TeX} = require(root+'/input/tex.js');
const {SVG} = require(root+'/output/svg.js');
const {liteAdaptor} = require(root+'/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require(root+'/handlers/html.js');
const {AllPackages} = require(root+'/input/tex/AllPackages.js');
const adaptor=liteAdaptor();RegisterHTMLHandler(adaptor);
const doc=mathjax.document('',{InputJax:new TeX({packages:AllPackages}),OutputJax:new SVG({fontCache:'none'})});
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const result=input.map(({expression,display})=>{
  const node=doc.convert(expression,{display,em:16,ex:8,containerWidth:1100});
  const outer=adaptor.outerHTML(node);
  if(outer.includes('data-mjx-error'))throw new Error('Invalid course math: '+expression);
  return outer.slice(outer.indexOf('<svg'),outer.lastIndexOf('</svg>')+6);
});
process.stdout.write(JSON.stringify(result));
