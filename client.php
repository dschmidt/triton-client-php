#!/usr/bin/env php
<?php

require_once 'vendor/autoload.php';

use Grpc\ChannelCredentials;
use Grpc\BaseStub;
use Inference\ModelInferRequest;
use Inference\ModelInferRequest\InferInputTensor;
use Inference\ModelInferRequest\InferRequestedOutputTensor;
use Inference\InferTensorContents;

class GRPCInferenceServiceClient extends BaseStub {

    public function __construct($hostname, $opts, $channel = null) {
        parent::__construct($hostname, $opts, $channel);
    }

    public function ModelStreamInfer(ModelInferRequest $argument,
                                    $metadata = [], $options = []) {
        return $this->_serverStreamRequest('/inference.GRPCInferenceService/ModelStreamInfer',
                                         $argument,
                                         ['\Inference\ModelStreamInferResponse', 'decode'],
                                         $metadata, $options);
    }
}

class TritonStreamingClient {
    private $client;
    private $tritonUrl;
    private $modelName;

    public function __construct() {
        $this->tritonUrl = getenv('TRITON_URL') ?: 'localhost:8001';
        $this->modelName = getenv('MODEL_NAME') ?: 'mistral-streaming';

        // Create gRPC client
        $this->client = new GRPCInferenceServiceClient(
            $this->tritonUrl,
            ['credentials' => ChannelCredentials::createInsecure()]
        );
    }

    public function streamInference($prompt, $maxTokens = 128) {
        $seenAny = false;

        // Prepare inputs
        $textInput = new InferInputTensor();
        $textInput->setName('text_input');
        $textInput->setDatatype('BYTES');
        $textInput->setShape([1, 1]);

        $textContents = new InferTensorContents();
        $textContents->setBytesContents([$prompt]);
        $textInput->setContents($textContents);

        $maxTokensInput = new InferInputTensor();
        $maxTokensInput->setName('max_tokens');
        $maxTokensInput->setDatatype('INT32');
        $maxTokensInput->setShape([1, 1]);

        $maxContents = new InferTensorContents();
        $maxContents->setIntContents([$maxTokens]);
        $maxTokensInput->setContents($maxContents);

        // Prepare outputs
        $textOutput = new InferRequestedOutputTensor();
        $textOutput->setName('text_output');

        $finalOutput = new InferRequestedOutputTensor();
        $finalOutput->setName('is_final');

        // Create request
        $request = new ModelInferRequest();
        $request->setModelName($this->modelName);
        $request->setId('req-1');
        $request->setInputs([$textInput, $maxTokensInput]);
        $request->setOutputs([$textOutput, $finalOutput]);

        try {
            // Start streaming inference
            $call = $this->client->ModelStreamInfer($request);

            // Process streaming responses
            foreach ($call->responses() as $response) {
                // Check for errors first
                $errorMessage = $response->getErrorMessage();
                if (!empty($errorMessage)) {
                    echo "[ERROR] " . $errorMessage . "\n";
                    continue;
                }

                $inferResponse = $response->getInferResponse();
                if (!$inferResponse) {
                    continue; // Skip if no infer response
                }
                $outputs = $inferResponse->getOutputs();

                $textChunk = '';
                $isFinal = false;

                // Use raw output contents if available
                $rawContents = $inferResponse->getRawOutputContents();
                if (!empty($rawContents)) {
                    // First raw content is typically the text output
                    if (isset($rawContents[0])) {
                        $textChunk = $rawContents[0];
                    }
                    // Check if we have is_final indicator in second raw content or detect end
                    if (count($rawContents) > 1 && isset($rawContents[1])) {
                        // Some models put boolean flags in second position
                        $isFinal = !empty(trim($rawContents[1]));
                    }
                } else {
                    // Fallback to structured output parsing
                    foreach ($outputs as $output) {
                        if ($output->getName() === 'text_output') {
                            $contents = $output->getContents();
                            if ($contents) {
                                $bytesContents = $contents->getBytesContents();
                                if (!empty($bytesContents)) {
                                    $textChunk = $bytesContents[0];
                                }
                            }
                        } elseif ($output->getName() === 'is_final') {
                            $contents = $output->getContents();
                            if ($contents) {
                                $boolContents = $contents->getBoolContents();
                                if (!empty($boolContents)) {
                                    $isFinal = $boolContents[0];
                                }
                            }
                        }
                    }
                }

                // Output the chunk (trim whitespace for cleaner output)
                if (!empty($textChunk)) {
                    echo $textChunk;
                    flush();
                    $seenAny = true;
                }

                // Check if this is the final chunk
                if ($isFinal) {
                    if ($seenAny) {
                        echo "\n";
                    }
                    break;
                }
            }

        } catch (Exception $e) {
            echo "\n[stream error] " . $e->getMessage() . "\n";
            return false;
        }

        return true;
    }
}

function main() {
    global $argv;

    $prompt = isset($argv[1]) ? $argv[1] : 'Sag hi in einem kurzen Satz.';
    $maxTokens = isset($argv[2]) ? intval($argv[2]) : 128;

    $client = new TritonStreamingClient();
    $client->streamInference($prompt, $maxTokens);
}

if (php_sapi_name() === 'cli') {
    main();
}
?>
