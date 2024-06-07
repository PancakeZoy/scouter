import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mygene

test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
ctrl_exp, pert_expr, gene_idx, bcode = next(iter(test_loader))

predict = model(gene_idx, ctrl_exp)
test_loss = gears_loss(predict, pert_expr, ctrl_exp, gene_idx)


sc.tl.rank_genes_groups(test_adata, 
                        groupby='condition',
                        reference = 'ctrl',
                        method='wilcoxon', 
                        n_genes=20)
df = sc.get.rank_genes_groups_df(test_adata, None)


all_genes = test_adata.var.index.tolist()
all_degs = df.names.unique()
mg = mygene.MyGeneInfo()
gene_info = mg.querymany(all_degs, scopes='ensembl.gene', fields='symbol', species='human')
ENSG_dict = {info['query']: info.get('symbol', 'N/A') for info in gene_info}

# Create a PdfPages object to save multiple pages
with PdfPages('/Users/pancake/Desktop/Admson_result.pdf') as pdf:
    for p in df.group.unique():
        # degs = df[df['group'] == p]['names'].values.tolist()
        degs = df[df['group']==p].nlargest(10, 'scores')['names'].values.tolist() + \
            df[df['group']==p].nsmallest(10, 'scores')['names'].values.tolist()
        gene_names = [ENSG_dict[g] for g in degs]
        degs_idx = [all_genes.index(g) for g in degs]
        pert_idx = int(test_adata[test_adata.obs['condition'] == p].obs.gene_index.unique())
        TestCells_pert_idx = (np.where(gene_idx.numpy() == pert_idx)[0]).tolist()
        
        test_mean = predict[np.ix_(TestCells_pert_idx, degs_idx)].mean(axis=0).detach().numpy()
        true_mean = pert_expr[np.ix_(TestCells_pert_idx, degs_idx)].mean(axis=0).numpy()
        ctrl_mean = ctrl_exp[:, degs_idx].mean(0).numpy()
        
        # Define the position of the bars on the x-axis
        ind = np.arange(len(degs))  
        width = 0.25
        
        # Create the bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plotting bars for ctrl_mean, test_mean, and true_mean
        bar1 = ax.bar(ind - width, ctrl_mean, width, label='Control', color='cornflowerblue')
        bar2 = ax.bar(ind, test_mean, width, label='Predict', color='forestgreen')
        bar3 = ax.bar(ind + width, true_mean, width, label='True', color='orange')
        
        # Add some text for labels, title, and custom x-axis tick labels
        ax.set_xlabel('Differentially Expressed Genes')
        ax.set_ylabel('Mean Expression Values')
        ax.set_title(f'Expression Values for Perturbation {p}')
        ax.set_xticks(ind)
        ax.set_xticklabels(gene_names, rotation=45)
        ax.legend()
        fig.subplots_adjust(bottom=0.3)
        ax.axvline(x=9.5, color='grey', linestyle='--')
        # Save the current figure to the PDF
        pdf.savefig(fig)
        
        # Close the figure to free up memory
        plt.close(fig)