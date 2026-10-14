import os
import sys
import torch
from grelu.resources import load_model
from grelu.data.dataset import VariantDataset
import pandas as pd

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    # Automatically force weights_only=False
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

def execute_platform_workflow(
    primary_tier: str,
    sub_task: str,
    repo_id: str,
    model_filename: str,
    input_files: dict, 
    output_dir: str,
):
  # 1. Hardware Detection (Safe for laptop shared CPU/GPU/MPS)
  if torch.cuda.is_available():
    device = torch.device('cuda')
  elif torch.backends.mps.is_available():
    device = torch.device('mps')
  else:
    device = torch.device('cpu')

  os.makedirs(output_dir, exist_ok=True)
  print(f'[*] Running [{primary_tier} -> {sub_task}] on device: {device}')
  print(f'[*] Loaded {len(input_files)} file input(s):')
  for key, path in input_files.items():
    print(f'    - {key}: {path}')

  # 2. Load Model Checkpoint from Hugging Face Model Zoo
  print(f'[*] Loading model [{model_filename}] from repo [{repo_id}]...')
  try:
    model = load_model(repo_id=repo_id, filename=model_filename)
    model.to(device)
    model.eval()
    print('[+] Model loaded successfully.')
  except Exception as e:
    print(f'[-] Error loading model: {e}')
    sys.exit(1)

  # 3. Dispatch to Category Handlers
  if primary_tier.upper() == 'VEP':
    handle_vep_workflow(sub_task, model, input_files, output_dir, device)
  elif primary_tier.upper() == 'DESIGN':
    handle_design_workflow(sub_task, model, input_files, output_dir, device)
  else:
    raise ValueError(f'Unknown primary tier: {primary_tier}')


def handle_vep_workflow(sub_task, model, input_path, output_dir,device):
  print(f'[*] Executing VEP module: {sub_task}')

  if sub_task == 'pathogenicity_disruption':
    
    vcf_path = input_path.get('vcf')
    fasta_path = input_path.get('fasta')
    
    if not vcf_path or not fasta_path:
          raise ValueError("Pathogenicity & Disruption requires both 'fasta' and 'vcf' files.")
    if not os.path.exists(vcf_path):
          raise FileNotFoundError(f'VCF file not found: {vcf_path}')
    if not os.path.exists(fasta_path):
          raise FileNotFoundError(f'Reference FASTA not found: {fasta_path}')

    print(f'    - Evaluating variants using VCF: {vcf_path}')
    print(f'    - Reference genome sequence: {fasta_path}')
    
    ensure_chrom_sizes_exist(fasta_path)
    
    # Load variant records (skipping comment headers)
    variants_df = pd.read_csv(vcf_path, comment='#', sep='\t', header=None)
    
    # DEBUG: Print what pandas actually sees
    print(f"    - Debug: Pandas detected {variants_df.shape[1]} column(s). Data preview:")
    print(variants_df.head())
    
    # Optional: Assign standard VCF column headers if they are missing
    if variants_df.shape[1] >= 5:
      col_names = ['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info']
      variants_df.columns = col_names[:variants_df.shape[1]]

      print(f'    - Loaded {len(variants_df)} variants. Setting up dataset...')

      # --- Active gReLU Model Prediction Block ---
      # 1. Initialize the dataset using the variants dataframe and reference FASTA
      
      '''
      actual code to be used in production
      seq_len = getattr(model, "data_params", {}).get("train", {}).get("seq_len", 1000)
      '''
      #used for testing
      seq_len = 30
      dataset = VariantDataset(variants=variants_df, genome=fasta_path, seq_len=seq_len)
        
      print(f'    - Running model predictions on device: {device}...')
      # 2. Execute model inference to obtain variant effect/disruption scores
      preds = model.predict_on_dataset(dataset, devices="cpu")
      # -------------------------------------------

      print('    - Model evaluation completed successfully.')
      
      # 3. Convert predictions to a standard NumPy array if it's a torch tensor
      if hasattr(preds, 'detach'):
          preds_array = preds.detach().cpu().numpy()
      elif isinstance(preds, torch.Tensor):
          preds_array = preds.cpu().numpy()
      else:
          preds_array = preds

      # 4. Save artifacts to output_dir
      os.makedirs(output_dir, exist_ok=True)
        
      # Save raw predictions matrix
      preds_file_path = os.path.join(output_dir, 'predictions.npy')
      import numpy as np
      np.save(preds_file_path, preds_array)
      print(f'    - Saved raw predictions to: {preds_file_path}')
      
      # Summarize execution metrics and prediction output details
      results_df = pd.DataFrame({
        'input_vcf': [vcf_path],
        'reference_fasta': [fasta_path],
        'variants_analyzed': [len(variants_df)],
        'analysis_type': ['Pathogenicity & Disruption'],
        'status': ['Completed successfully'],
        'output_predictions_shape': [str(preds.shape) if hasattr(preds, 'shape') else len(preds)]
      })
      
      summary_file_path = os.path.join(output_dir, 'summary_report.csv')
      results_df.to_csv(summary_file_path, index=False)
      print(f'    - Saved summary report to: {summary_file_path}')

      return results_df, preds

  elif sub_task == 'allele_specific_regulation':
    # Logic for allele-specific expression / regulatory differences
    pass

  elif sub_task == 'variant_prioritization':
    # Ranking variants based on predicted functional impact scores
    pass

  elif sub_task == 'multi_allelic_combinatorial':
    # Evaluating combinatorial effects of multiple mutations
    pass

  elif sub_task == 'saturation_mutagenesis':
    # Systematic in-silico mutation scanning across a genomic window
    pass

  # Save output artifacts (CSV / plots) to output_dir
  print('[+] VEP analysis complete.')


def handle_design_workflow(sub_task, model, input_path, output_dir):
  print(f'[*] Executing Sequence Design module: {sub_task}')

  if sub_task == 'synthetic_promoter_engineering':
    # Designing or optimizing promoter sequence elements for target expression
    pass

  elif sub_task == 'cell_specific_tuning':
    # Optimizing sequences for specific cell types or tissue contexts
    pass

  elif sub_task == 'directed_evolution':
    # In-Silico Mutagenesis (ISM) & motif insertion scanning
    pass

  elif sub_task == 'gradient_based_optimization':
    # Gradient-based sequence optimization (e.g., Ledidi optimization loop)
    pass

  elif sub_task == 'de_novo_generation':
    # De novo sequence generation and stress-testing against hallucination metrics
    pass

  # Save output sequence files (.fa) and score reports to output_dir
  print('[+] Design workflow complete.')

def ensure_chrom_sizes_exist(fasta_path):
    """Automatically generates a .sizes file from a FASTA file if it's missing."""
    sizes_path = fasta_path + ".sizes"
    if not os.path.exists(sizes_path):
        print(f"[*] Generating missing chromosome sizes file for {fasta_path}...")
        chrom_lengths = {}
        current_chrom = None
        current_len = 0
        
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_chrom:
                        chrom_lengths[current_chrom] = current_len
                    # Grab the chromosome name (everything after '>' up to the first space)
                    current_chrom = line[1:].split()[0]
                    current_len = 0
                else:
                    current_len += len(line)
            if current_chrom:
                chrom_lengths[current_chrom] = current_len
                
        # Write out the .sizes file
        with open(sizes_path, 'w') as f:
            for chrom, length in chrom_lengths.items():
                f.write(f"{chrom}\t{length}\n")
        print(f"[+] Created {sizes_path}")


if __name__ == '__main__':
  # Quick direct test example using a python dictionary dictionary representation
  # In production, Flask will build this dictionary dynamically from user uploads.
  sample_files = {
      'fasta': 'mock-test-data/test_ref.fasta',
      'vcf': 'mock-test-data/test_variants.vcf',
  }

  execute_platform_workflow(
      primary_tier='VEP',
      sub_task='pathogenicity_disruption',
      repo_id='Genentech/human-atac-catlas-model',
      model_filename='model.ckpt',
      input_files=sample_files,
      output_dir='outputs',
  )
  