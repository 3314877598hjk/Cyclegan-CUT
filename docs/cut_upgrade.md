# CUT Upgrade Notes

This project now supports a one-way CUT backend alongside the original
CycleGAN baseline. The new model keeps the existing ResNet generator and
self-attention option, removes the inverse generator dependency, and trains
with:

- adversarial loss from a single PatchGAN discriminator
- PatchNCE contrastive loss over multiple generator feature layers
- Sobel edge consistency loss for structure preservation

## Main Commands

Train the improved CUT model:

```bash
python train.py \
  --dataroot ./datasets/maps \
  --name map2vector_cut \
  --model cut \
  --CUT_mode CUT \
  --direction AtoB \
  --lambda_edge 1.0
```

Train the lighter FastCUT variant:

```bash
python train.py \
  --dataroot ./datasets/maps \
  --name map2vector_fastcut \
  --model cut \
  --CUT_mode FastCUT \
  --direction AtoB \
  --lambda_edge 1.0
```

Run inference with a trained CUT checkpoint:

```bash
python test.py \
  --dataroot ./datasets/maps/testA \
  --name map2vector_cut \
  --model cut \
  --num_test 50
```

## Ablation Switches

- `--no_attention`: disables the self-attention block in the generator.
- `--no_edge_loss`: disables Sobel structure preservation.
- `--no_nce_idt`: disables identity-domain PatchNCE in CUT mode.
- `--CUT_mode FastCUT`: uses FastCUT-style defaults, including stronger NCE
  weight and no identity-domain NCE unless `--nce_idt` is supplied.

Suggested paper groups:

- `B0`: `--model cycle_gan`
- `B1`: `--model cut --CUT_mode CUT`
- `B2`: `--model cut --CUT_mode FastCUT`
- `B3`: `--model cut --CUT_mode CUT` with attention and edge loss enabled

Suggested ablations:

- `A0`: `--model cut --no_attention --no_edge_loss`
- `A1`: `--model cut --no_attention`
- `A2`: `--model cut --no_edge_loss`
- `A3`: `--model cut`
