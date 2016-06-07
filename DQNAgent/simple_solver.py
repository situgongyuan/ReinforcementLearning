import optimizer as optim

def step(params,grads,update_rule):
	if hasattr(optim,update_rule):
		update_rule = getattr(optim, update_rule)
		for key in params.keys():
			new_x,config = update_rule(params[key],grads[key])
			params[key] = new_x

