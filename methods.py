"""
Different explanation methods from the Integrated-Gradient Optimized Saliency map methods.
© copyright 2024 Bytedance Ltd. and/or its affiliates.
Modified from Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""

from torch.autograd import Variable
from methods_helper import *

def iGOS_pp(
        args,
        model,
        model_name, 
        init_mask,
        image,
        baseline,
        label,
        size=32,
        iterations=15,
        ig_iter=20,
        L1=1,
        L2=1,
        L3=20,
        lr=1000,
        opt='LS',
        softmax=True,
        **kwargs):

    def regularization_loss(image, masks):
        return L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1), \
               L3 * bilateral_tv_norm(image, masks, resolution, tv_beta=2, sigma=0.01), \
               L2 * torch.sum((1 - masks)**2, dim=[1, 2, 3])

    def ins_loss_function(up_masks, indices, noise=True):
        losses = -interval_score(
                    model,
                    model_name, 
                    baseline[indices],
                    image[indices],
                    label[indices],
                    up_masks,
                    ig_iter,
                    noise
                    )
        return losses.sum(dim=1).view(-1)

    def del_loss_function(up_masks, indices, noise=True):
        losses = interval_score(
                    model,
                    model_name, 
                    image[indices],
                    baseline[indices],
                    label[indices],
                    up_masks,
                    ig_iter,
                    noise,
                    )
        return losses.sum(dim=1).view(-1)

    def loss_function(up_masks, masks, indices):
        loss = del_loss_function(up_masks[:, 0], indices)
        loss += ins_loss_function(up_masks[:, 1], indices)
        loss += del_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        loss += ins_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        return loss + regularization_loss(image[indices], masks[:, 0] * masks[:, 1])

    masks_del = torch.ones((1, 1, size, size), dtype=torch.float32, device='cuda')
    masks_del = masks_del * init_mask.cuda()
    masks_del = Variable(masks_del, requires_grad=True)
    masks_ins = torch.ones((image.shape[0], 1, size, size), dtype=torch.float32, device='cuda')
    masks_ins = masks_ins * init_mask.cuda()
    masks_ins = Variable(masks_ins, requires_grad=True)
    prompt = kwargs.get('prompt', None)
    image_size = kwargs.get('image_size', None)
    positions = kwargs.get('positions', None)
    resolution = kwargs.get('resolution', None)

    if opt == 'NAG':
        cita_d=torch.zeros(1).cuda()
        cita_i=torch.zeros(1).cuda()
    
    prompt = kwargs.get('prompt', None)
    image_size = kwargs.get('image_size', None)
    positions = kwargs.get('positions', None)

    losses_del, losses_ins, losses_l1, losses_tv, losses_l2, losses_comb_del, losses_comb_ins = [], [], [], [], [], [], []
    for i in range(iterations):
        up_masks1 = upscale(masks_del, image, resolution)
        up_masks2 = upscale(masks_ins, image, resolution)

        # Compute the integrated gradient for the combined mask, optimized for deletion
        loss_comb_del = integrated_gradient(args, model, model_name, image, baseline, label, up_masks1 * up_masks2, ig_iter, prompt=prompt, image_size=image_size, positions=positions)
        total_grads1 = masks_del.grad.clone()
        total_grads2 = masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Compute the integrated gradient for the combined mask, optimized for insertion
        loss_comb_ins = integrated_gradient(args, model, model_name, baseline, image, label, up_masks1 * up_masks2, ig_iter, prompt=prompt, image_size=image_size, positions=positions)
        total_grads1 -= masks_del.grad.clone()  # Negative because insertion loss is 1 - score.
        total_grads2 -= masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Compute the integrated gradient for the deletion mask
        loss_del = integrated_gradient(args, model, model_name, image, baseline, label, up_masks1, ig_iter, prompt=prompt, image_size=image_size, positions=positions)
        total_grads1 += masks_del.grad.clone()
        masks_del.grad.zero_()

        # Compute the integrated graident for the insertion mask
        loss_ins = integrated_gradient(args, model, model_name, baseline, image, label, up_masks2, ig_iter, prompt=prompt, image_size=image_size, positions=positions)
        total_grads2 -= masks_ins.grad.clone()
        masks_ins.grad.zero_()

        # Average them to balance out the terms with the regularization terms
        total_grads1 /= 2
        total_grads2 /= 2

        # Computer regularization for combined masks
        L2 = exp_decay(args.L2, i, args.gamma)
        loss_l1, loss_tv, loss_l2 = regularization_loss(image, masks_del * masks_ins)
        losses = loss_l1 + loss_tv + loss_l2
        losses.sum().backward()
        total_grads1 += masks_del.grad.clone()
        total_grads2 += masks_ins.grad.clone()

        if opt == 'LS':
            masks = torch.cat((masks_del.unsqueeze(1), masks_ins.unsqueeze(1)), 1)
            total_grads = torch.cat((total_grads1.unsqueeze(1), total_grads2.unsqueeze(1)), 1)
            lrs = line_search(masks, total_grads, loss_function, lr)
            masks_del.data -= total_grads1 * lrs
            masks_ins.data -= total_grads2 * lrs
        
        if opt == 'NAG':
            e = i / (i + args.momentum)
            cita_d_p = cita_d
            cita_i_p = cita_i
            cita_d = masks_del.data - lr * total_grads1
            cita_i = masks_ins.data - lr * total_grads2
            masks_del.data = cita_d + e * (cita_d - cita_d_p)
            masks_ins.data = cita_i + e * (cita_i - cita_i_p)

        masks_del.grad.zero_()
        masks_ins.grad.zero_()
        masks_del.data.clamp_(0,1)
        masks_ins.data.clamp_(0,1)

        losses_del.append(loss_del)
        losses_ins.append(loss_ins)
        losses_comb_del.append(loss_comb_del)
        losses_comb_ins.append(loss_comb_ins)
        losses_l1.append(loss_l1.item())
        losses_tv.append(loss_tv.item())
        losses_l2.append(loss_l2.item())
        print(f'iteration: {i} lr: {lr:.4f} loss_comb_del: {loss_comb_del:.4f}, loss_comb_ins: {loss_comb_ins:.4f}, loss_del: {loss_del:.4f}, loss_ins: {loss_ins:.4f}, loss_l1: {loss_l1.item():.4f}, loss_tv: {loss_tv.item():.4f}, loss_l2: {loss_l2.item():.4f}')

    return masks_del * masks_ins, losses_del, losses_ins, losses_l1, losses_tv, losses_l2, losses_comb_del, losses_comb_ins


def iGOS_p(
        args,
        model,
        model_name, 
        init_mask,
        image,
        baseline,
        label,
        size=28,
        iterations=15,
        ig_iter=20,
        L1=1,
        L2=1,
        L3=20,
        lr=1000,
        opt='LS',
        softmax=True,
        **kwargs):

    def regularization_loss(image, masks, resolution):
        return L1 * torch.mean(torch.abs(1-masks).view(masks.shape[0],-1), dim=1), \
               L3 * bilateral_tv_norm(image, masks, resolution, tv_beta=2, sigma=0.01), \
               L2 * torch.sum((1 - masks)**2, dim=[1, 2, 3])

    def loss_function(up_masks, masks, indices, noise=True):
        losses = -interval_score(
                    model,
                    model_name, 
                    baseline[indices],
                    image[indices],
                    label[indices],
                    up_masks,
                    ig_iter,
                    noise
                    )
        losses += interval_score(
                    model,
                    model_name, 
                    image[indices],
                    baseline[indices],
                    label[indices],
                    up_masks,
                    ig_iter,
                    noise
                    )
        return losses.sum(dim=1).view(-1) + regularization_loss(masks)
    
    masks = torch.ones((1,1,size,size), dtype=torch.float32, device='cuda')
    masks = masks * init_mask.cuda()
    masks = Variable(masks, requires_grad=True)
    prompt = kwargs.get('prompt', None)
    image_size = kwargs.get('image_size', None)
    positions = kwargs.get('positions', None)
    resolution = kwargs.get('resolution', None)

    if opt == 'NAG':
        cita=torch.zeros(1).cuda()

    losses_del, losses_ins, losses_l1, losses_tv, losses_l2 = [], [], [], [], []
    for i in range(iterations):
        total_grads = torch.zeros_like(masks).cuda()
        up_masks = upscale(masks, image, resolution)

        loss_del = integrated_gradient(args, model, model_name, image, baseline, label, up_masks, ig_iter, prompt=prompt, image_size=image_size, positions=positions)
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        loss_ins = integrated_gradient(args, model, model_name, baseline, image, label, up_masks, ig_iter, prompt=prompt, image_size=image_size, positions=positions)
        total_grads += -masks.grad.clone()
        masks.grad.zero_()

        L2 = exp_decay(args.L2, i, args.gamma)
        loss_l1, loss_tv, loss_l2 = regularization_loss(image, masks, resolution)
        losses = loss_l1 + loss_tv + loss_l2
        losses.sum().backward()
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        if opt == 'LS':
            lrs = line_search(masks, total_grads, loss_function, lr)
            masks.data -= total_grads * lrs
        
        if opt == 'NAG':
            e = i / (i + args.momentum) 
            cita_p = cita
            cita = masks.data - lr * total_grads
            masks.data = cita + e * (cita - cita_p)

        losses_del.append(loss_del)
        losses_ins.append(loss_ins)
        losses_l1.append(loss_l1.item())
        losses_tv.append(loss_tv.item())
        losses_l2.append(loss_l2.item())

        print(f'iteration: {i} lr: {lr:.4f} loss_del: {loss_del:.4f}, loss_ins: {loss_ins:.4f}, loss_l1: {loss_l1.item():.4f}, loss_tv: {loss_tv.item():.4f}, loss_l2: {loss_l2.item():.4f}')
        masks.grad.zero_()
        masks.data.clamp_(0, 1)

    return masks, losses_del, losses_ins, losses_l1, losses_tv, losses_l2, None, None
