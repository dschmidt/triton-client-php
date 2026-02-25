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
    private $tritonModel;

    public function __construct() {
        $this->tritonUrl = getenv('TRITON_URL') ?: 'localhost:8001';
        $this->modelName = getenv('MODEL_NAME') ?: 'mistral-7b-instruct-v0.3';
        $this->tritonModel = getenv('TRITON_MODEL') ?: 'streaming';

        // Create gRPC client
        $this->client = new GRPCInferenceServiceClient(
            $this->tritonUrl,
            ['credentials' => ChannelCredentials::createInsecure()]
        );
    }

    public function streamInference($conversation, $maxTokens = 128) {
        $seenAny = false;
        $lastChannel = 'content';

        // Prepare inputs
        $textInput = new InferInputTensor();
        $textInput->setName('conversation');
        $textInput->setDatatype('BYTES');
        $textInput->setShape([1, 1]);

        $textContents = new InferTensorContents();
        $textContents->setBytesContents([
            json_encode($conversation, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES)
        ]);
        $textInput->setContents($textContents);

        // Model name — which vLLM backend model to use
        $modelNameInput = new InferInputTensor();
        $modelNameInput->setName('model_name');
        $modelNameInput->setDatatype('BYTES');
        $modelNameInput->setShape([1, 1]);

        $modelNameContents = new InferTensorContents();
        $modelNameContents->setBytesContents([$this->modelName]);
        $modelNameInput->setContents($modelNameContents);

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

        $channelOutput = new InferRequestedOutputTensor();
        $channelOutput->setName('channel');

        $finalOutput = new InferRequestedOutputTensor();
        $finalOutput->setName('is_final');

        // Create request — always target the "streaming" wrapper model
        $request = new ModelInferRequest();
        $request->setModelName($this->tritonModel);
        $request->setId('req-1');
        $request->setInputs([$textInput, $modelNameInput, $maxTokensInput]);
        $request->setOutputs([$textOutput, $channelOutput, $finalOutput]);

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
                $channel = 'content';
                $isFinal = false;

                // Use raw output contents if available
                // Triton raw BYTES format: 4-byte LE length prefix + string bytes per element
                // Output order matches config.pbtxt: text_output[0], channel[1], is_final[2]
                $rawContents = $inferResponse->getRawOutputContents();
                if (!empty($rawContents)) {
                    if (isset($rawContents[0]) && strlen($rawContents[0]) >= 4) {
                        $len = unpack('V', substr($rawContents[0], 0, 4))[1];
                        $textChunk = substr($rawContents[0], 4, $len);
                    }
                    if (isset($rawContents[1]) && strlen($rawContents[1]) >= 4) {
                        $len = unpack('V', substr($rawContents[1], 0, 4))[1];
                        $channel = substr($rawContents[1], 4, $len);
                    }
                    if (isset($rawContents[2]) && strlen($rawContents[2]) >= 1) {
                        // is_final is a single BOOL byte
                        $isFinal = ord($rawContents[2][0]) !== 0;
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
                        } elseif ($output->getName() === 'channel') {
                            $contents = $output->getContents();
                            if ($contents) {
                                $bytesContents = $contents->getBytesContents();
                                if (!empty($bytesContents)) {
                                    $channel = $bytesContents[0];
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

                // Output the chunk with ANSI formatting per channel
                if (!empty($textChunk)) {
                    if ($channel !== 'final' && $channel !== 'content') {
                        // Non-final channels (analysis, commentary) in grey
                        if ($lastChannel === 'final' || $lastChannel === 'content') {
                            echo "\033[90m--- {$channel} ---\033[0m\n";
                            echo "\033[90m";
                        } elseif ($lastChannel !== $channel) {
                            echo "\033[0m\n\033[90m--- {$channel} ---\033[0m\n";
                            echo "\033[90m";
                        }
                        echo $textChunk;
                    } else {
                        if ($lastChannel !== 'final' && $lastChannel !== 'content') {
                            echo "\033[0m\n\033[90m--- {$channel} ---\033[0m\n";
                        }
                        echo $textChunk;
                    }
                    flush();
                    $seenAny = true;
                    $lastChannel = $channel;
                }

                // Check if this is the final chunk
                if ($isFinal) {
                    if ($lastChannel !== 'final' && $lastChannel !== 'content') {
                        echo "\033[0m";
                    }
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
    $conversation = [
        ["role" => "system", "content" => "Antworte auf Deutsch."],
        ["role" => "user", "content" => $prompt]
    ];
    $maxTokens = isset($argv[2]) ? intval($argv[2]) : 128;

    $client = new TritonStreamingClient();
    $client->streamInference($conversation, $maxTokens);
}

if (php_sapi_name() === 'cli') {
    main();
}
