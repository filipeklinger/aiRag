// Required NuGet packages:
// - LangChain
// - LangChain.Providers.Ollama
// - LangChain.Databases.Sqlite
// - Ollama

using LangChain.Databases;
using LangChain.Databases.Sqlite;
using LangChain.DocumentLoaders;
using LangChain.Providers.Ollama;
using LangChain.Extensions;
using Ollama;
using System.Text;

namespace src.Services;


public class LlmSummaryService
{
    private readonly OllamaProvider _provider;
    private readonly OllamaEmbeddingModel _embeddingModel;
    private readonly OllamaChatModel _llm;
    private readonly SqLiteVectorDatabase _vectorDatabase;
    private IVectorCollection _vectorCollection;

    private static bool _isExampleCreated = false;

    public LlmSummaryService()
    {
        _provider = new OllamaProvider();

        _embeddingModel = new OllamaEmbeddingModel(_provider, id: "all-minilm");
        _llm = new OllamaChatModel(_provider, id: "llama3");

        _vectorDatabase = new SqLiteVectorDatabase(dataSource: "vectors.db");
        CreateDatabaseAsync().Wait();
    }

    private async Task CreateDatabaseAsync()
    {
        // Cria o banco de dados e a coleção
        await _vectorDatabase.CreateCollectionAsync(
            collectionName: "summaries",
            dimensions: 384
        );

        _vectorCollection = await _vectorDatabase.GetCollectionAsync("summaries");
        
    }

    public async Task AddExampleAsync(string textoOriginal, string resumoEsperado)
    {
        string combinado = $"Texto: {textoOriginal}\nResumo: {resumoEsperado}";
        var document = new Document(combinado,
            metadata: new Dictionary<string, object>
            {
                { "textoOriginal", textoOriginal },
                { "resumoEsperado", resumoEsperado }
            }
        );
        await _vectorCollection.AddDocumentsAsync(_embeddingModel,[document]);
    }

    private async Task CreateExampleAsync()
    {
        if(_isExampleCreated)
            return;
        System.Console.WriteLine("Creating examples...");
        var exampleText1 = "A água é essencial para a vida na Terra. Ela cobre cerca de 71% da superfície do planeta e é vital para todos os seres vivos. Sem água, a vida como conhecemos não seria possível.";
        var exampleSummary1 = "A água é essencial para a vida e cobre 71% da Terra.";

        var exampleText2 = "As florestas tropicais são importantes para o equilíbrio ecológico. Elas abrigam uma enorme biodiversidade e ajudam a regular o clima global. Além disso, são uma fonte crucial de oxigênio para o planeta.";
        var exampleSummary2 = "Florestas tropicais são cruciais para biodiversidade, clima e oxigênio.";

        var exampleText3 = "A energia solar é uma fonte renovável e limpa. Ela pode ser utilizada para gerar eletricidade e aquecer ambientes. Com o avanço da tecnologia, a energia solar está se tornando mais acessível.";
        var exampleSummary3 = "Energia solar é limpa, renovável e cada vez mais acessível.";

        await AddExampleAsync(exampleText1, exampleSummary1);
        await AddExampleAsync(exampleText2, exampleSummary2);
        await AddExampleAsync(exampleText3, exampleSummary3);
        //vamos adicionar mais exemplos agora in english
        var exampleText4 = "The sun is the center of our solar system. It provides light and heat to the planets, including Earth. Without the sun, life on Earth would not be possible.";
        var exampleSummary4 = "The sun is the center of the solar system, providing light and heat essential for life on Earth.";
        var exampleText5 = "The internet has revolutionized communication. It allows people to connect instantly across the globe. Social media platforms have changed how we interact and share information.";
        var exampleSummary5 = "The internet revolutionized communication, enabling instant global connections and changing interactions through social media.";
        var exampleText6 = "Artificial intelligence is transforming industries. It can analyze data, automate tasks, and improve decision-making. AI is being used in healthcare, finance, and many other fields.";
        var exampleSummary6 = "AI is transforming industries by analyzing data, automating tasks, and improving decision-making in various fields.";
        await AddExampleAsync(exampleText4, exampleSummary4);
        await AddExampleAsync(exampleText5, exampleSummary5);
        await AddExampleAsync(exampleText6, exampleSummary6);

        //vamos adicionar mais exemplos agora in spanish
        var exampleText7 = "La energía eólica es una fuente renovable de energía. Se genera a partir del viento y se utiliza para producir electricidad. La energía eólica es limpia y sostenible.";
        var exampleSummary7 = "La energía eólica es una fuente renovable y limpia de electricidad generada por el viento.";
        var exampleText8 = "La biodiversidad es crucial para el equilibrio de los ecosistemas. Incluye la variedad de especies, genes y ecosistemas. La pérdida de biodiversidad puede tener graves consecuencias para el medio ambiente.";
        var exampleSummary8 = "La biodiversidad es esencial para el equilibrio de los ecosistemas y su pérdida puede ser perjudicial.";
        var exampleText9 = "La inteligencia artificial está cambiando la forma en que trabajamos. Puede analizar grandes cantidades de datos y ayudar en la toma de decisiones. La IA se utiliza en diversas industrias, desde la salud hasta las finanzas.";
        var exampleSummary9 = "La IA está transformando el trabajo al analizar datos y ayudar en decisiones en diversas industrias.";
        await AddExampleAsync(exampleText7, exampleSummary7);
        await AddExampleAsync(exampleText8, exampleSummary8);
        await AddExampleAsync(exampleText9, exampleSummary9);
        _isExampleCreated = true;
    }

    public async Task<string> SummarizeAsync(string texto)
    {
        await CreateExampleAsync();

        var similarDocuments = await _vectorCollection.GetSimilarDocuments(
            _embeddingModel,
            texto,
            amount: 3
        );

        var context = similarDocuments.AsString();

        var prompt = $"""
Use os seguintes exemplos de contexto para gerar um resumo.
Se não houver contexto suficiente, diga "não sei".
Mantenha a resposta curta e clara.

{context}

Texto: {texto}
Resumo:
""";

        var answer = _llm.GenerateAsync(prompt);
        Console.WriteLine($"LLM answer: {answer}");
        var finalMessage = new StringBuilder();
        await foreach (var message in answer)
        {
            finalMessage.AppendLine(message.LastMessageContent);
        }
        return finalMessage.ToString();
    }
}
