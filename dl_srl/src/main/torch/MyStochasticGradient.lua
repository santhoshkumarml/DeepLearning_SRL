--
-- User: santhosh
-- Date: 11/30/15
--

local MyStochasticGradient = torch.class('MyStochasticGradient')

function MyStochasticGradient:__init(module, criterion)
    self.learningRate = 0.01
    self.learningRateDecay = 0
    self.maxIteration = 25
    self.shuffleIndices = true
    self.module = module
    self.criterion = criterion
    self.verbose = true
end

function MyStochasticGradient:train(dataset, hookExample)
    local iteration = 1
    local currentLearningRate = self.learningRate
    local module = self.module
    local criterion = self.criterion

    local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
    if not self.shuffleIndices then
        for t = 1,dataset:size() do
            shuffledIndices[t] = t
        end
    end

    while true do
        local currentError = 0
        for t = 1,dataset:size() do
            local example = dataset[shuffledIndices[t]]
            local input = example[1]
            local target = example[2]

            currentError = currentError + criterion:forward(module:forward(input), target)

            module:updateGradInput(input, criterion:updateGradInput(module.output, target))
            module:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)

            if hookExample then
                hookExample(shuffledIndices[t])
            end
        end

        currentError = currentError / dataset:size()

        if self.hookIteration then
            self.hookIteration(self, iteration, currentError)
        end

        iteration = iteration + 1
        currentLearningRate = self.learningRate/(1+iteration*self.learningRateDecay)
        if self.maxIteration > 0 and iteration > self.maxIteration and self.verbose then
            print("# My StochasticGradient - training error = " .. currentError)
            break
        end
    end
end

