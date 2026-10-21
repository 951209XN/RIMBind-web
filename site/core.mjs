export function parseCSV(text) {
  if (text.length > 2 * 1024 * 1024) throw new Error('CSV exceeds 2 MiB.');
  const lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/);
  if(lines.length<2)throw new Error('CSV has no residue rows.');
  const headers=lines.shift().split(',').map(x=>x.trim());
  const required=['residue_index','amino_acid','sscp_probability','rimbind_probability'];
  if(required.some(x=>!headers.includes(x)))throw new Error('Required CSV columns are missing.');
  if(new Set(headers).size!==headers.length)throw new Error('Duplicate CSV column names.');
  if(lines.length>600)throw new Error('This initial viewer supports at most 600 residues.');
  return lines.map((line,i)=>{
    const cells=line.split(',');if(cells.length!==headers.length)throw new Error('CSV row width mismatch. Use plain unquoted scalar columns.');
    const r=Object.fromEntries(headers.map((h,j)=>[h,cells[j].trim()]));
    if(Number(r.residue_index)!==i+1)throw new Error('Residue indices must be unique, contiguous 1-based model positions.');
    if(!/^[ACDEFGHIKLMNPQRSTVWY]$/.test(r.amino_acid))throw new Error('Unsupported amino-acid code.');
    for(const k of required.slice(2)){if(r[k]===''||!Number.isFinite(Number(r[k]))||Number(r[k])<0||Number(r[k])>1)throw new Error('Probabilities must be finite numbers in [0, 1].');r[k]=Number(r[k]);}
    r.residue_index=i+1;r.difference=r.rimbind_probability-r.sscp_probability;return r;
  });
}
const AA={ALA:'A',ARG:'R',ASN:'N',ASP:'D',CYS:'C',GLN:'Q',GLU:'E',GLY:'G',HIS:'H',ILE:'I',LEU:'L',LYS:'K',MET:'M',PHE:'F',PRO:'P',SER:'S',THR:'T',TRP:'W',TYR:'Y',VAL:'V'};
export function inspectPDB(text,chain){
  if(new TextEncoder().encode(text).length>2097152)throw new Error('PDB exceeds 2 MiB.');
  if(!/^[A-Za-z0-9]$/.test(chain))throw new Error('Select one alphanumeric PDB chain identifier.');
  const seen=new Map();let models=0;
  for(const line of text.split(/\r?\n/)){
    if(line.startsWith('MODEL ')){models++;if(models>1)break;}if(line.startsWith('ENDMDL'))break;
    if(line.startsWith('HETATM')&&line.slice(21,22)===chain&&line.slice(17,20)==='MSE')throw new Error('MSE needs a separately validated conversion; not silently converted.');
    if(!line.startsWith('ATOM  ')||line.slice(21,22)!==chain)continue;
    const name=line.slice(17,20).trim();if(!AA[name])throw new Error('Selected chain contains a nonstandard or non-protein ATOM residue.');
    const alt=line.slice(16,17);if(alt!==' '&&alt!=='A')continue;
    const number=Number(line.slice(22,26)),insertion=line.slice(26,27).trim();
    if(!Number.isInteger(number))throw new Error('Invalid residue number.');
    const key=`${chain}|${number}|${insertion}`;
    if(!seen.has(key))seen.set(key,{key,chain,number,insertion,aa:AA[name],ca:false});
    const r=seen.get(key);if(r.aa!==AA[name])throw new Error('Conflicting residue identity.');
    if(line.slice(12,16).trim()==='CA'){
      if(r.ca)throw new Error('Ambiguous alternate C-alpha atom.');
      if([30,38,46].some(i=>!Number.isFinite(Number(line.slice(i,i+8)))||!line.slice(i,i+8).trim()))throw new Error('Invalid C-alpha coordinates.');
      r.ca=true;
    }
  }
  const residues=[...seen.values()];if(!residues.length)throw new Error('No protein residues found for this chain.');
  if(residues.length>600)throw new Error('Initial service limit is 600 observed residues.');
  if(residues.some(x=>!x.ca))throw new Error('Every residue needs a C-alpha coordinate.');
  return {chain,residues,sequence:residues.map(x=>x.aa).join('')};
}
export function assertMapping(rows,pdb){if(rows.map(r=>r.amino_acid).join('')!==pdb.sequence)throw new Error('Prediction CSV does not match the observed PDB-chain sequence. Scores were not mapped to this structure.');}
export function scoreColor(value,difference=false){
  if(!Number.isFinite(value))return '#e6ebea';
  let colors= difference?[[50,105,158],[238,241,237],[178,59,53]]:[[233,239,237],[218,219,173],[231,180,126],[185,56,53]];
  const t=Math.max(0,Math.min(1,difference?(value+1)/2:value))*(colors.length-1);const i=Math.min(Math.floor(t),colors.length-2),f=t-i;
  return '#'+colors[i].map((v,k)=>Math.round(v+(colors[i+1][k]-v)*f).toString(16).padStart(2,'0')).join('');
}
export function mapPDB(text,pdb,rows){
  assertMapping(rows,pdb);const values=new Map(pdb.residues.map((r,i)=>[r.key,rows[i].rimbind_probability]));
  return text.split(/\r?\n/).map(line=>{
    if(!line.startsWith('ATOM  '))return line;
    const key=`${line.slice(21,22)}|${Number(line.slice(22,26))}|${line.slice(26,27).trim()}`;
    const p=values.get(key);const b=p===undefined?-1:p;
    return line.padEnd(80,' ').slice(0,60)+b.toFixed(2).padStart(6,' ')+line.padEnd(80,' ').slice(66);
  }).join('\n');
}
