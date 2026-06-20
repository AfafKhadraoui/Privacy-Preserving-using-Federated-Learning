# MobileStyleGAN Inversion Explained

This document explains how the MobileStyleGAN-based inversion attack in `src/attacks/model_inversion.py` works, with emphasis on the latent space, the `w` vector, the loss terms, and the optimization loop.

## 1. What MobileStyleGAN Is

MobileStyleGAN is a compact face generator. In this project it is used as a decoder that turns a latent code into a face image.

The attack does not try to classify a person. Instead, it tries to find a latent code that produces an image whose embedding matches the target face embedding from the victim.

In simple terms:

- input: a latent code
- generator: MobileStyleGAN
- output: a face image
- attacker goal: make the generated image look like the target face in embedding space

## 2. The Main Pieces in the Code

The attack has three main parts:

- `load_mobilestylegan(device)`: loads the MobileStyleGAN checkpoint and builds a lightweight generator wrapper
- `run_mobilestylegan_inversion_attack(...)`: optimizes the latent representation
- `get_target_embedding(model, client_dir)`: loads a real face crop and computes the target embedding once

## 3. What the Latent Means

In generative models, a latent is a hidden numeric representation that controls the generated image.

You can think of it as a compressed description of a face:

- face shape
- pose
- lighting
- expression
- identity-related structure

The latent is not the image itself. It is a set of numbers that the generator turns into an image.

## 4. What the `w` Vector Is

MobileStyleGAN uses a style-based latent representation. In this code, the key latent is `w`.

`w` is the style vector produced by the mapping network, and it is easier to optimize than raw noise.

The code starts from:

```python
w_avg = generator.style_mean.detach().clone()
w_plus = w_avg.unsqueeze(1).repeat(1, 23, 1).detach().clone().requires_grad_(True)
```

That means:

- `w_avg` is the generator’s average style vector
- `w_plus` is the W+ version of that vector
- the code repeats the same style vector across 23 layers

## 5. What W+ Means

Regular `W` uses one style vector for the whole generator.

`W+` gives each synthesis layer its own style vector.

So instead of one vector controlling everything, MobileStyleGAN can receive a different style input per layer.

Why this matters:

- `W` is simpler and more constrained
- `W+` is more flexible
- `W+` usually reconstructs details better because each layer can adjust separately

In this project, the attack uses W+ because it gives a stronger reconstruction result.

## 6. Why the Attack Starts From the Average Style

The starting point is not random noise.

It starts from the generator’s average style:

```python
w_avg = generator.style_mean.detach().clone()
```

That is a learned summary of the latent space center.

This is useful because:

- it is a stable starting point
- it already represents a realistic face prior
- optimization converges faster than random initialization

So the attack begins from a plausible face-like latent, then refines it toward the victim embedding.

## 7. How the Generator Produces an Image

Inside `MobileStyleGANLite`, the forward pass works like this:

1. the mapping network converts the latent input into style codes
2. the student synthesis network turns those styles into an image
3. the resulting image is returned as `img`

In code:

```python
if style is None:
    style = self.mapping_net(var)
return self.student(style)["img"]
```

So the generator is really learning how to decode style vectors into a face.

## 8. What the Target Embedding Is

The target embedding is the face representation of the victim image produced by the face recognition model.

The code first loads a crop from `client_dir`, then runs it through the face model:

```python
embedding = model(tensor)
```

This embedding is the target the optimizer tries to match.

Important point:

- the crop is used to compute the target embedding
- the crop is not used directly as the reconstruction target image
- the optimization only sees the embedding target, not the pixel image

## 9. The Loss Function

The attack optimizes the latent using several loss terms.

### 9.1 Identity Loss

This is the main loss.

It measures how close the generated face embedding is to the victim embedding:

```python
identity_loss = 1.0 - F.cosine_similarity(generated_emb, target_embedding.detach(), dim=1).mean()
```

If cosine similarity is high, the loss is low.

This means the generated image is becoming more identity-consistent with the victim.

### 9.2 W+ Regularization

This keeps the optimized latent close to the average face prior:

```python
w_reg = torch.mean((w_plus - w_avg.unsqueeze(1)).pow(2))
```

This prevents the latent from drifting too far into unrealistic regions.

### 9.3 Total Variation Loss

This penalizes sharp pixel-to-pixel changes:

```python
tv = total_variation_loss(img_s)
```

Total variation helps reduce noise and checkerboard artifacts.

### 9.4 Symmetry Loss

This encourages a front-facing face by making the image similar to its horizontal flip:

```python
sym = symmetry_loss(img_s)
```

Faces are often roughly symmetric, so this is a useful prior.

### 9.5 Combined Loss

The total loss is:

```python
loss = 20.0 * identity_loss + 0.5 * w_reg + 0.00005 * tv + 0.05 * sym
```

Interpretation:

- identity loss dominates the objective
- W+ regularization keeps the latent realistic
- TV loss removes noisy artifacts
- symmetry loss encourages a plausible face pose

## 10. How Optimization Works

The latent `w_plus` is treated as a learnable tensor:

```python
w_plus = ...requires_grad_(True)
optimizer = torch.optim.Adam([w_plus], lr=lr)
```

Then for each iteration:

1. generate an image from `w_plus`
2. compute the face embedding of the generated image
3. compute the loss against the target embedding
4. backpropagate gradients into `w_plus`
5. update `w_plus` with Adam

This is standard gradient descent in latent space.

### Why Adam

Adam is used because it adapts the learning rate per parameter and tends to work better than plain SGD for latent optimization.

## 11. Why Noise Is Added During Early Iterations

The code adds small noise to the generated image during the early phase:

```python
if i < iterations * 0.8:
    noise = torch.randn_like(img_s) * 0.01
    img_proc = img_s + noise
else:
    img_proc = img_s
```

This helps the optimizer explore the space and avoid getting stuck too early in a bad local minimum.

Later, the code removes the noise so the final image can refine cleanly.

## 12. Why the Image Is Resized to 160x160

The face model expects 160x160 inputs, so the generated image is resized before embedding extraction:

```python
synth_img = F.interpolate(img_proc, size=(160, 160), mode='bilinear', align_corners=False)
```

This is important because the embedding model and the generator may not use the same output resolution.

## 13. What the Final Output Is

After all iterations, the code generates the final image from the optimized W+ latent and saves it:

```python
final_image_batch = generator(style=w_plus.detach()).detach()
```

The saved result is the reconstructed face approximation.

The function returns:

- the final reconstructed image
- the final loss value

## 14. Full Intuition

The attack is basically this:

1. take the victim face embedding
2. start from the generator’s average face latent
3. render a face from that latent
4. compare the rendered face embedding to the victim embedding
5. move the latent so the generated face becomes more similar
6. repeat until the latent produces a close reconstruction

So the generator acts like a face prior, and the embedding loss tells the optimizer how to move inside that prior.

## 15. Short Answer

The MobileStyleGAN attack starts from the generator’s average style latent `style_mean`, expanded into W+.

It does not start from random noise.

It does not start from the average of the cropped face images.

The cropped faces are only used to compute the target embedding.

## 16. Related Files

- [src/attacks/model_inversion.py](../src/attacks/model_inversion.py)
- [src/model/face_model.py](../src/model/face_model.py)
- [scratch_check_ckpt.py](../scratch_check_ckpt.py)
