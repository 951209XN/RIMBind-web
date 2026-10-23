import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
import {parseCSV,inspectPDB,assertMapping,mapPDB,scoreColor} from '../site/core.mjs';
const csv=fs.readFileSync(new URL('../site/examples/example.csv',import.meta.url),'utf8');
const pdb=fs.readFileSync(new URL('../site/examples/example.pdb',import.meta.url),'utf8');
test('Example has 53 finite probability pairs with strict sequence mapping',()=>{const rows=parseCSV(csv),p=inspectPDB(pdb,'A');assert.equal(rows.length,53);assertMapping(rows,p);});
test('Reject duplicate or non-contiguous model residue positions',()=>assert.throws(()=>parseCSV(csv.replace('2,L,','1,L,'))));
test('Reject NaN probabilities',()=>assert.throws(()=>parseCSV(csv.replace('0.133550614','NaN'))));
test('Reject mismatched structure and score sequences',()=>assert.throws(()=>assertMapping(parseCSV(csv),{sequence:'AAAA'})));
test('Wrong chain is not silently substituted',()=>assert.throws(()=>inspectPDB(pdb,'Z')));
test('Mapped B fields represent probability not percent',()=>{const r=parseCSV(csv),p=inspectPDB(pdb,'A');const out=mapPDB(pdb,p,r).split('\n').find(x=>x.startsWith('ATOM  '));assert.equal(Number(out.slice(60,66)),.13);});
test('Difference color zero is neutral',()=>assert.equal(scoreColor(0,true),'#eef1ed'));
const html=fs.readFileSync(new URL('../site/index.html',import.meta.url),'utf8');
const app=fs.readFileSync(new URL('../site/app.js',import.meta.url),'utf8');
const config=JSON.parse(fs.readFileSync(new URL('../site/config.json',import.meta.url),'utf8'));
test('External predictor CTA has the intended host and no file forwarding',()=>{
  const tag=html.match(/<a\s[^>]*id="open-predictor"[^>]*>/)?.[0];
  assert.ok(tag);assert.ok(tag.includes('href="https://huggingface.co/spaces/XuNing985/RIMBind-web"'));
  assert.ok(tag.includes('target="_blank"'));assert.ok(tag.includes('noopener noreferrer'));
  assert.equal(config.space_url,'https://huggingface.co/spaces/XuNing985/RIMBind-web');
  assert.equal(config.direct_upload_from_pages,false);assert.equal(config.live_prediction_enabled,false);
  assert.ok(!app.includes('FormData'));assert.ok(!/method\s*:\s*['"]POST/.test(app));
});
test('Static explorer does not masquerade as an in-page prediction backend',()=>{
  assert.match(html,/<button id="predict" disabled>/);
  assert.match(html,/not a live uptime indicator/);
  assert.ok(!html.includes('New-protein inference is not yet available.'));
  assert.match(html,/No file is transferred automatically/);
});
test('Privacy and quota limitations are visible',()=>{
  for(const text of ['Hugging Face infrastructure','not a platform-wide deletion guarantee','quotas','restart','one hour'])assert.ok(html.includes(text));
});
test('Frontend rejects blank probabilities and oversize CSV',()=>{
  assert.throws(()=>parseCSV(csv.replace('0.133550614','')));
  assert.throws(()=>parseCSV(' '.repeat(2097153)));
});
test('Frontend rejects nonfinite CA and missing CA coordinates',()=>{
  const ca=pdb.split('\n').find(l=>l.startsWith('ATOM  ')&&l.slice(12,16).trim()==='CA');
  assert.throws(()=>inspectPDB(pdb.replace(ca,ca.slice(0,30)+'     NaN'+ca.slice(38)),'A'));
  assert.throws(()=>inspectPDB(pdb.split('\n').filter(l=>l.slice(12,16).trim()!=='CA').join('\n'),'A'));
});
test('Public guide and acceptance notes exist without fabricated performance claims',()=>{
  const guide=fs.readFileSync(new URL('../deploy/USER_GUIDE.md',import.meta.url),'utf8');
  const status=fs.readFileSync(new URL('../deploy/RESEARCH_BETA_STATUS_20261023.md',import.meta.url),'utf8');
  assert.match(guide,/NO_TEMPLATE_COUNT/);assert.match(guide,/first model/);
  assert.match(status,/public_service_acceptance_completed:false/);
  assert.match(status,/failed, retained/);assert.match(status,/mocked/);
});
