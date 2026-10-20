from lm_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parts=corpus();result={'lesson':'101','device':'cpu','split_sizes':{k:len(v) for k,v in parts.items()},'models':{}}
    for label,merges in [('character',0),('bpe',24)]:
        tok=Tokenizer(parts['train'],merges);model,r=train(parts['train'],parts['validation'],tok,width=24,steps=120,seed=7)
        r['test']=evaluate(model,parts['test'],tok);r['vocab']=len(tok.symbols);result['models'][label]=r
        if label=='bpe':
            tok.save();example=parts['test'][0];ids=tok.encode(example)
            result['alignment']={'text':example,'input':[tok.symbols[i] for i in ids[:-1]],'label':[tok.symbols[i] for i in ids[1:]],'padding_loss':'ignored','eos_loss':'included'}
            x,y=batch([example],tok)
            with torch.no_grad():
                original=model(x);changed=x.clone();changed[0,-1]=(changed[0,-1]+1)%len(tok.symbols);altered=model(changed)
                result['future_invariance_max_difference']=float((original[:,:-1]-altered[:,:-1]).abs().max())
    (ROOT/'outputs/results.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    for label,r in result['models'].items():ax[0].plot([h['tokens'] for h in r['history']],[h['validation_nll'] for h in r['history']],'-o',label=label)
    ax[0].set(xlabel='Processed target tokens',ylabel='Validation nats/token',title='Different units: do not rank raw NLL');ax[0].legend()
    names=list(result['models']);ax[1].bar(names,[result['models'][n]['test']['bits_per_utf8_byte'] for n in names],color=['#2877aa','#36a078']);ax[1].set(ylabel='Test bits / UTF-8 byte',title='Shared text denominator (EOS included)')
    fig.tight_layout();fig.savefig(ROOT/'figures/results.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
    for i,(top,bottom) in enumerate(zip(['BOS','the ','red ','cat'],['the ','red ','cat','EOS'])):
        ax.text(i*.25+.1,.75,top,ha='center',bbox={'boxstyle':'round','facecolor':'#dceefd'});ax.annotate('',xy=(i*.25+.1,.32),xytext=(i*.25+.1,.61),arrowprops={'arrowstyle':'->'});ax.text(i*.25+.1,.2,bottom,ha='center',bbox={'boxstyle':'round','facecolor':'#def5e7'})
    ax.set_title('Teacher forcing: position t predicts the NEXT token; no future input');fig.tight_layout();fig.savefig(ROOT/'figures/mechanism.png',dpi=160);plt.close(fig)
    print(json.dumps({'models':{k:{'test_ppl':v['test']['ppl'],'test_bits_per_byte':v['test']['bits_per_utf8_byte']} for k,v in result['models'].items()},'future_invariance':result['future_invariance_max_difference']},indent=2))
if __name__=='__main__':
    main()
    import runpy
    runpy.run_path(str(ROOT/'draw_mechanism.py'))
