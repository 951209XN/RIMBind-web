import {parseCSV,inspectPDB,assertMapping,scoreColor,mapPDB} from './core.mjs';
const $=id=>document.getElementById(id);let state={rows:[],pdb:'',mapping:null,kind:'none',task:'dna'},viewer=null;
function setText(id,value){$(id).textContent=value;}
function error(target,e){setText(target,e.message||String(e));}
function plot(){
  const canvas=$('probability-chart'),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(300,Math.round(rect.width*dpr));canvas.height=Math.max(200,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.scale(dpr,dpr);
  const W=canvas.width/dpr,H=canvas.height/dpr,left=32,right=10,top=16,bottom=27,w=W-left-right,h=H-top-bottom;
  c.clearRect(0,0,W,H);c.font='10px Arial';c.lineWidth=1;c.textAlign='right';
  for(const v of [0,.25,.5,.75,1]){const y=top+h*(1-v);c.strokeStyle='#e6ece9';c.beginPath();c.moveTo(left,y);c.lineTo(W-right,y);c.stroke();c.fillStyle='#70827f';c.fillText(String(v),left-6,y+3);}
  c.setLineDash([4,4]);c.strokeStyle='#b89d6a';c.beginPath();c.moveTo(left,top+h/2);c.lineTo(W-right,top+h/2);c.stroke();c.setLineDash([]);
  const n=state.rows.length;if(!n)return;
  for(const [field,color,width] of [['sscp_probability','#a2b4b1',1.1],['rimbind_probability','#087c7b',1.8]]){c.strokeStyle=color;c.lineWidth=width;c.beginPath();state.rows.forEach((r,i)=>{const x=left+i/Math.max(1,n-1)*w,y=top+(1-r[field])*h;i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();}
  c.fillStyle='#70827f';c.textAlign='center';[1,Math.round(n/2),n].forEach(i=>c.fillText(String(i),left+(i-1)/Math.max(1,n-1)*w,H-8));
}
function drawMolecule(){
  const lib=window.$3Dmol;
  if(!lib){$('molecule').replaceChildren();const p=document.createElement('p');p.className='viewer-fallback';p.textContent='3D viewer unavailable. Sequence map, scores and downloads remain available.';$('molecule').append(p);return;}
  if(!viewer){$('molecule').replaceChildren();viewer=lib.createViewer($('molecule'),{backgroundColor:'white',antialias:true});}
  viewer.clear();if(!state.pdb){viewer.render();return;}
  const mode=$('colour-mode').value,lookup=new Map();
  if(state.mapping&&state.rows.length){try{assertMapping(state.rows,state.mapping);state.mapping.residues.forEach((r,i)=>lookup.set(r.key,state.rows[i][mode]));}catch(e){setText('input-report',e.message);}}
  viewer.addModel(state.pdb,'pdb');viewer.setStyle({},{cartoon:{colorfunc:a=>scoreColor(lookup.get(`${a.chain}|${a.resi}|${a.icode||''}`),mode==='difference'),thickness:.55}});
  viewer.zoomTo();viewer.render();
}
function draw(){
  setText('residue-count',String(state.rows.length));setText('binding-count',String(state.rows.filter(r=>r.rimbind_probability>=.5).length));
  const map=$('sequence-map');map.replaceChildren();for(const r of state.rows){const b=document.createElement('button');b.className='residue';b.textContent=r.amino_acid;b.style.background=scoreColor(r.rimbind_probability);const description=`${r.residue_index} ${r.amino_acid} · SSCP ${r.sscp_probability.toFixed(3)} · RIMBind ${r.rimbind_probability.toFixed(3)}`;b.title=description;b.setAttribute('aria-label',description);b.addEventListener('pointerenter',()=>setText('residue-detail',description));b.addEventListener('focus',()=>setText('residue-detail',description));map.append(b);}
  $('download-csv').disabled=!state.rows.length;$('download-json').disabled=!state.rows.length;$('download-pdb').disabled=!state.rows.length||!state.mapping||state.rows.map(r=>r.amino_acid).join('')!==state.mapping.sequence;
  plot();drawMolecule();
}
async function example(){
  try{const [p,c]=await Promise.all([fetch('examples/example.pdb'),fetch('examples/example.csv')]);if(!p.ok||!c.ok)throw Error('Example files are missing.');const pdb=await p.text(),csv=await c.text();state={rows:parseCSV(csv),pdb,mapping:inspectPDB(pdb,'A'),kind:'precomputed_example',task:'dna'};assertMapping(state.rows,state.mapping);$('task').value='dna';$('chain').value='A';setText('result-title','DNA example · 6JHE chain A');setText('result-subtitle','Precomputed, label-free demonstration of five-fold SSCP and DNA FULL fusion.');setText('result-kind','PRECOMPUTED EXAMPLE');setText('file-name','Documented 6JHE chain A example');setText('input-report','Example loaded. No inference request was made.');draw();}catch(e){error('input-report',e);}
}
$('example').addEventListener('click',example);
$('pdb-file').addEventListener('change',()=>setText('file-name',$('pdb-file').files[0]?.name||'Single protein chain · up to 2 MiB'));
$('inspect-input').addEventListener('click',async()=>{try{const f=$('pdb-file').files[0];if(!f)throw Error('Choose a PDB file first.');if(f.size>2097152)throw Error('PDB exceeds 2 MiB.');const pdb=await f.text(),mapping=inspectPDB(pdb,$('chain').value);state={rows:[],pdb,mapping,kind:'local_structure_only',task:$('task').value};setText('result-title','Local structure check · no prediction');setText('result-subtitle','This structure has no computed RIMBind scores. Previous example scores have been cleared.');setText('result-kind','NO PREDICTIONS');setText('input-report',`Chain ${mapping.chain}: ${mapping.sequence.length} observed residues. Local check passed; model prediction was not run.`);draw();}catch(e){error('input-report',e);}});
$('task').addEventListener('change',()=>{state={rows:[],pdb:'',mapping:null,kind:'none',task:$('task').value};setText('result-title',`${state.task.toUpperCase()} task · awaiting verified predictions`);setText('result-subtitle','Changing the task does not convert existing scores or run the model.');setText('result-kind','NO PREDICTIONS');draw();});
$('csv-file').addEventListener('change',async()=>{try{const f=$('csv-file').files[0];if(!f)return;const rows=parseCSV(await f.text());if(state.mapping)assertMapping(rows,state.mapping);state={...state,rows,kind:'user_supplied_predictions',task:$('task').value};setText('result-title','Your existing probability table');setText('result-subtitle','Model provenance is user-supplied and has not been independently verified. No new probabilities were calculated.');setText('result-kind','USER-SUPPLIED SCORES');setText('csv-report',`${rows.length} rows validated for visualization.`);draw();}catch(e){error('csv-report',e);}});
$('colour-mode').addEventListener('change',()=>{const delta=$('colour-mode').value==='difference';$('colour-scale').style.background=delta?'linear-gradient(90deg,#32699e,#eef1ed,#b23b35)':'';drawMolecule();});
$('reset-view').addEventListener('click',()=>{viewer?.zoomTo();viewer?.render();});
function download(name,data,mime){const url=URL.createObjectURL(new Blob([data],{type:mime})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('download-csv').addEventListener('click',()=>{const lines=['residue_index,amino_acid,sscp_probability,rimbind_probability'];state.rows.forEach(r=>lines.push([r.residue_index,r.amino_acid,r.sscp_probability,r.rimbind_probability].join(',')));download('rimbind_probabilities.csv',lines.join('\n'),'text/csv');});
$('download-pdb').addEventListener('click',()=>{try{download('rimbind_probability_B.pdb',mapPDB(state.pdb,state.mapping,state.rows),'chemical/x-pdb');}catch(e){error('input-report',e);}});
$('download-json').addEventListener('click',()=>download('rimbind_result_provenance.json',JSON.stringify({source_kind:state.kind,task:state.task,prediction_generated_in_this_session:false,model_version:state.kind==='precomputed_example'?'DNA FULL frozen example':'not independently verified',chain:state.mapping?.chain??null,residue_mapping:state.mapping?.residues??null,threshold:.5,metrics_computed:false},null,2),'application/json'));
window.addEventListener('resize',()=>{plot();viewer?.resize();});
fetch('config.json').then(r=>r.json()).then(c=>{setText('service-status',c.service_status);/* Never enable live inference solely from a public frontend flag. */}).catch(()=>{});
example();
