using Microsoft.KernelMemory;
using Microsoft.KernelMemory.AI.Ollama;
using Microsoft.KernelMemory.AI;
using System.Text;
using Microsoft.KernelMemory.Diagnostics;

namespace src.Services;



public class LlmSummaryService
{
    private readonly IKernelMemory _memory;

    public LlmSummaryService()
    {
        var logLevel = LogLevel.Warning;
        SensitiveDataLogger.Enabled = true;

        var config = new OllamaConfig
        {
            Endpoint = "http://localhost:11434",
            TextModel = new OllamaModelConfig("gemma3:1b")
            {
                MaxTokenTotal = 125000,
                Seed = 42,
                TopK = 7
            },
            EmbeddingModel = new OllamaModelConfig("nomic-embed-text:latest")
            {
                MaxTokenTotal = 2048
            }
        };

        _memory = new KernelMemoryBuilder()
           .WithOllamaTextGeneration(config, new CL100KTokenizer())
           .WithOllamaTextEmbeddingGeneration(config, new CL100KTokenizer())
           .Configure(builder => builder.Services.AddLogging(l =>
           {
               l.SetMinimumLevel(logLevel);
               l.AddSimpleConsole(c => c.SingleLine = true);
           }))
           .Build();
    }

    public async Task<string> SummarizeAsync(string textoNovo)
    {

        var answer = await _memory.AskAsync("Resuma este texto: " + textoNovo);

        var contextText = string.Join("\n---\n",
            answer.RelevantSources
                .SelectMany(c => c.Partitions)
                .Select(p => p.Text));

        var finalPrompt = $"""
        Baseando-se nos seguintes exemplos de resumo:
        {contextText}

        Agora, resuma este novo texto:
        {textoNovo}
        """;

        var resumo = await _memory.AskAsync(finalPrompt);
        return resumo.Result;

    }

    // Opcional: método para indexar seus exemplos
    public async Task ImportarExemploAsync(string id, string textoOriginal, string resumoEsperado)
    {
        string combinado = $"""
        Texto:
        {textoOriginal}

        Resumo:
        {resumoEsperado}
        """;
        await _memory.ImportTextAsync(
            text: combinado,
            documentId: id,
            tags: new TagCollection { ["tipo"] = ["exemplo-resumo"] }
        );

    }
}

