import torch
from torch.autograd import Variable

from geo_tf_score import geo_tf_interval_nll, geo_tf_interval_nll_backward
from methods_helper import bilateral_tv_norm, exp_decay, upscale


def geo_tf_regularization(args, image, masks, step):
    loss_area = args.L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1).sum()
    loss_tv = args.L3 * bilateral_tv_norm(image, masks, None, tv_beta=2, sigma=0.01).sum()
    l2_weight = exp_decay(args.L2, step, args.gamma)
    loss_l2 = l2_weight * torch.sum((1 - masks) ** 2, dim=[1, 2, 3]).sum()
    return loss_area, loss_tv, loss_l2


def geo_tf_del_nll(args, adapter, model, prompt_inputs, image, baseline, up_mask, reference):
    return geo_tf_interval_nll(
        args=args,
        adapter=adapter,
        model=model,
        prompt_inputs=prompt_inputs,
        image=image,
        baseline=baseline,
        up_mask=up_mask,
        reference=reference,
    )


def geo_tf_ins_nll(args, adapter, model, prompt_inputs, image, baseline, up_mask, reference):
    return geo_tf_interval_nll(
        args=args,
        adapter=adapter,
        model=model,
        prompt_inputs=prompt_inputs,
        image=baseline,
        baseline=image,
        up_mask=up_mask,
        reference=reference,
    )


def geo_tf_del_nll_backward(args, adapter, model, prompt_inputs, image, baseline, make_up_mask, reference):
    return geo_tf_interval_nll_backward(
        args=args,
        adapter=adapter,
        model=model,
        prompt_inputs=prompt_inputs,
        image=image,
        baseline=baseline,
        make_up_mask=make_up_mask,
        reference=reference,
        grad_scale=-1.0,
    )


def geo_tf_ins_nll_backward(args, adapter, model, prompt_inputs, image, baseline, make_up_mask, reference):
    return geo_tf_interval_nll_backward(
        args=args,
        adapter=adapter,
        model=model,
        prompt_inputs=prompt_inputs,
        image=baseline,
        baseline=image,
        make_up_mask=make_up_mask,
        reference=reference,
        grad_scale=1.0,
    )


def geo_tf_line_search_igos(args, adapter, model, image, baseline, prompt_inputs, reference, masks, total_grads, step):
    def loss_fn(candidate_masks):
        up_mask = upscale(candidate_masks, image, None)
        del_nll = geo_tf_del_nll(args, adapter, model, prompt_inputs, image, baseline, up_mask, reference)
        ins_nll = geo_tf_ins_nll(args, adapter, model, prompt_inputs, image, baseline, up_mask, reference)
        loss_area, loss_tv, loss_l2 = geo_tf_regularization(args, image, candidate_masks, step)
        return -del_nll + ins_nll + loss_area + loss_tv + loss_l2

    with torch.no_grad():
        alpha = torch.tensor(float(args.lr), dtype=masks.dtype, device=masks.device)
        base_loss = loss_fn(masks)
        target_slope = -0.0001 * (total_grads ** 2).sum()
        while alpha >= 0.00001:
            new_masks = torch.clamp(masks - alpha.view(1, 1, 1, 1) * total_grads, 0, 1)
            new_loss = loss_fn(new_masks)
            if new_loss <= base_loss + alpha * target_slope:
                break
            alpha = alpha * 0.2
    return alpha


def geo_tf_line_search_igos_pp(
    args,
    adapter,
    model,
    image,
    baseline,
    prompt_inputs,
    reference,
    masks_del,
    masks_ins,
    total_grads_del,
    total_grads_ins,
    step,
):
    def loss_fn(candidate_del, candidate_ins):
        up_masks_del = upscale(candidate_del, image, None)
        up_masks_ins = upscale(candidate_ins, image, None)
        up_masks_comb = up_masks_del * up_masks_ins
        comb_del_nll = geo_tf_del_nll(args, adapter, model, prompt_inputs, image, baseline, up_masks_comb, reference)
        comb_ins_nll = geo_tf_ins_nll(args, adapter, model, prompt_inputs, image, baseline, up_masks_comb, reference)
        del_nll = geo_tf_del_nll(args, adapter, model, prompt_inputs, image, baseline, up_masks_del, reference)
        ins_nll = geo_tf_ins_nll(args, adapter, model, prompt_inputs, image, baseline, up_masks_ins, reference)
        loss_area, loss_tv, loss_l2 = geo_tf_regularization(args, image, candidate_del * candidate_ins, step)
        return -comb_del_nll + comb_ins_nll - del_nll + ins_nll + loss_area + loss_tv + loss_l2

    with torch.no_grad():
        alpha = torch.tensor(float(args.lr), dtype=masks_del.dtype, device=masks_del.device)
        base_loss = loss_fn(masks_del, masks_ins)
        target_slope = -0.0001 * ((total_grads_del ** 2).sum() + (total_grads_ins ** 2).sum())
        while alpha >= 0.00001:
            new_masks_del = torch.clamp(masks_del - alpha.view(1, 1, 1, 1) * total_grads_del, 0, 1)
            new_masks_ins = torch.clamp(masks_ins - alpha.view(1, 1, 1, 1) * total_grads_ins, 0, 1)
            new_loss = loss_fn(new_masks_del, new_masks_ins)
            if new_loss <= base_loss + alpha * target_slope:
                break
            alpha = alpha * 0.2
    return alpha


def geo_tf_igos_mask_optimization(
    args,
    adapter,
    model,
    init_mask,
    image,
    baseline,
    prompt_inputs,
    reference,
):
    device = image.device
    masks = torch.ones((1, 1, args.size, args.size), dtype=torch.float32, device=device)
    masks = Variable(masks * init_mask.to(device), requires_grad=True)
    cita = torch.zeros_like(masks.data)
    history = []

    for step in range(args.iterations):
        if masks.grad is not None:
            masks.grad.zero_()

        del_nll = geo_tf_del_nll_backward(
            args,
            adapter,
            model,
            prompt_inputs,
            image,
            baseline,
            lambda: upscale(masks, image, None),
            reference,
        )
        total_grads = masks.grad.clone()
        masks.grad.zero_()

        ins_nll = geo_tf_ins_nll_backward(
            args,
            adapter,
            model,
            prompt_inputs,
            image,
            baseline,
            lambda: upscale(masks, image, None),
            reference,
        )
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        loss_area, loss_tv, loss_l2 = geo_tf_regularization(args, image, masks, step)
        loss_regularization = loss_area + loss_tv + loss_l2
        loss_regularization.backward()
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        if args.opt == "LS":
            alpha = geo_tf_line_search_igos(
                args, adapter, model, image, baseline, prompt_inputs, reference, masks, total_grads, step
            )
            masks.data -= alpha * total_grads

        if args.opt == "NAG":
            e = step / (step + args.momentum)
            cita_prev = cita
            cita = masks.data - args.lr * total_grads
            masks.data = cita + e * (cita - cita_prev)

        masks.grad.zero_()
        masks.data.clamp_(0, 1)

        objective = -del_nll + ins_nll + loss_regularization
        deletion_area = float(torch.mean(1 - masks).detach().cpu())
        record = {
            "step": int(step),
            "objective": float(objective.detach().cpu()),
            "geo_tf_nll": float(del_nll.detach().cpu()),
            "geo_tf_del_nll": float(del_nll.detach().cpu()),
            "geo_tf_ins_nll": float(ins_nll.detach().cpu()),
            "area": float(loss_area.detach().cpu()),
            "tv": float(loss_tv.detach().cpu()),
            "l2": float(loss_l2.detach().cpu()),
            "deletion_area": deletion_area,
        }
        history.append(record)
        print(
            "step={step} objective={objective:.6f} del_nll={geo_tf_del_nll:.6f} "
            "ins_nll={geo_tf_ins_nll:.6f} del_area={deletion_area:.4f}".format(**record)
        )

    return masks.detach(), history


def geo_tf_igos_pp_mask_optimization(
    args,
    adapter,
    model,
    init_mask,
    image,
    baseline,
    prompt_inputs,
    reference,
):
    device = image.device
    masks_del = torch.ones((1, 1, args.size, args.size), dtype=torch.float32, device=device)
    masks_del = Variable(masks_del * init_mask.to(device), requires_grad=True)
    masks_ins = torch.ones((image.shape[0], 1, args.size, args.size), dtype=torch.float32, device=device)
    masks_ins = Variable(masks_ins * init_mask.to(device), requires_grad=True)
    cita_d = torch.zeros_like(masks_del.data)
    cita_i = torch.zeros_like(masks_ins.data)
    history = []

    for step in range(args.iterations):
        if masks_del.grad is not None:
            masks_del.grad.zero_()
        if masks_ins.grad is not None:
            masks_ins.grad.zero_()

        comb_del_nll = geo_tf_del_nll_backward(
            args,
            adapter,
            model,
            prompt_inputs,
            image,
            baseline,
            lambda: upscale(masks_del, image, None) * upscale(masks_ins, image, None),
            reference,
        )
        total_grads_del = masks_del.grad.clone()
        total_grads_ins = masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        comb_ins_nll = geo_tf_ins_nll_backward(
            args,
            adapter,
            model,
            prompt_inputs,
            image,
            baseline,
            lambda: upscale(masks_del, image, None) * upscale(masks_ins, image, None),
            reference,
        )
        total_grads_del += masks_del.grad.clone()
        total_grads_ins += masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        del_nll = geo_tf_del_nll_backward(
            args,
            adapter,
            model,
            prompt_inputs,
            image,
            baseline,
            lambda: upscale(masks_del, image, None),
            reference,
        )
        total_grads_del += masks_del.grad.clone()
        masks_del.grad.zero_()

        ins_nll = geo_tf_ins_nll_backward(
            args,
            adapter,
            model,
            prompt_inputs,
            image,
            baseline,
            lambda: upscale(masks_ins, image, None),
            reference,
        )
        total_grads_ins += masks_ins.grad.clone()
        masks_ins.grad.zero_()

        total_grads_del /= 2
        total_grads_ins /= 2

        masks_comb = masks_del * masks_ins
        loss_area, loss_tv, loss_l2 = geo_tf_regularization(args, image, masks_comb, step)
        loss_regularization = loss_area + loss_tv + loss_l2
        loss_regularization.backward()
        total_grads_del += masks_del.grad.clone()
        total_grads_ins += masks_ins.grad.clone()

        if args.opt == "LS":
            alpha = geo_tf_line_search_igos_pp(
                args,
                adapter,
                model,
                image,
                baseline,
                prompt_inputs,
                reference,
                masks_del,
                masks_ins,
                total_grads_del,
                total_grads_ins,
                step,
            )
            masks_del.data -= alpha * total_grads_del
            masks_ins.data -= alpha * total_grads_ins

        if args.opt == "NAG":
            e = step / (step + args.momentum)
            cita_d_prev = cita_d
            cita_i_prev = cita_i
            cita_d = masks_del.data - args.lr * total_grads_del
            cita_i = masks_ins.data - args.lr * total_grads_ins
            masks_del.data = cita_d + e * (cita_d - cita_d_prev)
            masks_ins.data = cita_i + e * (cita_i - cita_i_prev)

        masks_del.grad.zero_()
        masks_ins.grad.zero_()
        masks_del.data.clamp_(0, 1)
        masks_ins.data.clamp_(0, 1)

        masks_comb = masks_del * masks_ins
        objective = -comb_del_nll + comb_ins_nll - del_nll + ins_nll + loss_regularization
        deletion_area = float(torch.mean(1 - masks_comb).detach().cpu())
        record = {
            "step": int(step),
            "objective": float(objective.detach().cpu()),
            "geo_tf_comb_del_nll": float(comb_del_nll.detach().cpu()),
            "geo_tf_comb_ins_nll": float(comb_ins_nll.detach().cpu()),
            "geo_tf_del_nll": float(del_nll.detach().cpu()),
            "geo_tf_ins_nll": float(ins_nll.detach().cpu()),
            "area": float(loss_area.detach().cpu()),
            "tv": float(loss_tv.detach().cpu()),
            "l2": float(loss_l2.detach().cpu()),
            "deletion_area": deletion_area,
        }
        history.append(record)
        print(
            "step={step} objective={objective:.6f} comb_del_nll={geo_tf_comb_del_nll:.6f} "
            "comb_ins_nll={geo_tf_comb_ins_nll:.6f} del_nll={geo_tf_del_nll:.6f} "
            "ins_nll={geo_tf_ins_nll:.6f} del_area={deletion_area:.4f}".format(**record)
        )

    return (masks_del * masks_ins).detach(), history
