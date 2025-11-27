// api-client.js - Cliente API configurable para comunicarse con el backend
import axios from "axios";

class APIClient {
  constructor() {
    // Configuración desde variables de entorno
    const useMock = import.meta.env.VITE_USE_MOCK === "true";
    this.baseURL = useMock
      ? import.meta.env.VITE_MOCK_API_URL || "http://localhost:3001"
      : import.meta.env.VITE_API_URL || "http://localhost:8000";

    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    console.log(`API Client initialized with base URL: ${this.baseURL}`);
  }

  /**
   * Transcribe audio a texto usando Azure Speech o servicio configurado
   * @param {Blob} audioBlob - Archivo de audio a transcribir
   * @returns {Promise<string>} - Texto transcrito
   */
  async transcribeAudio(audioBlob) {
    const formData = new FormData();
    formData.append("audio", audioBlob, "audio.webm");

    try {
      const response = await this.client.post("/api/v1/transcribe", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      return response.data.transcription || response.data.text || "";
    } catch (error) {
      console.error("Error transcribiendo audio:", error);
      throw new Error(
        error.response?.data?.detail || "Error en la transcripción"
      );
    }
  }

  /**
   * Clasifica una imagen usando un modelo específico
   * @param {Blob} imageBlob - Imagen a clasificar
   * @param {string} model - Nombre del modelo (opcional)
   * @returns {Promise<Object>} - Resultado de la clasificación
   */
  async classifyImage(imageBlob, model = "modelo_perros_resnet") {
    const formData = new FormData();
    formData.append("image", imageBlob, "image.jpg");

    try {
      const response = await this.client.post(
        `/api/v1/classify/image?model=${model}`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      return response.data;
    } catch (error) {
      console.error("Error clasificando imagen:", error);
      throw new Error(
        error.response?.data?.detail || "Error en la clasificación de imagen"
      );
    }
  }

  /**
   * Reconoce rostro y detecta emociones en una imagen
   * @param {Blob} imageBlob - Imagen con el rostro
   * @returns {Promise<Object>} - Datos del rostro y emociones
   */
  async recognizeFace(imageBlob) {
    const formData = new FormData();
    formData.append("image", imageBlob, "frame.jpg");

    try {
      const response = await this.client.post(
        "/api/v1/face/sentiment",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      return response.data;
    } catch (error) {
      console.error("Error reconociendo rostro:", error);
      throw new Error(
        error.response?.data?.detail || "Error en el reconocimiento facial"
      );
    }
  }

  /**
   * Procesa un comando de texto con el modelo ML
   * @param {string|Object} command - Comando o payload estructurado
   * @returns {Promise<string>} - Respuesta del modelo
   */
  async processCommand(command) {
    try {
      const payload = typeof command === "string" ? { text: command } : command;

      const response = await this.client.post(
        "/api/v1/command/execute",
        payload
      );

      // Normalizar la respuesta
      if (response.data.response) return response.data.response;
      if (response.data.result) return response.data.result;
      if (response.data.transcription) return response.data.transcription;

      return JSON.stringify(response.data);
    } catch (error) {
      console.error("Error procesando comando:", error);
      throw new Error(
        error.response?.data?.detail || "Error procesando comando"
      );
    }
  }

  /**
   * Calcula BMI o BodyFat
   * @param {Object} params - { height, weight, age }
   * @returns {Promise<Object>} - Resultado del cálculo
   */
  async calculateBMI(params) {
    try {
      const response = await this.client.post("/api/v1/bmi", params);
      return response.data;
    } catch (error) {
      console.error("Error calculando BMI:", error);
      throw new Error(error.response?.data?.detail || "Error calculando BMI");
    }
  }

  /**
   * Verifica el estado del sistema
   * @returns {Promise<Object>} - Estado de salud del sistema
   */
  async healthCheck() {
    try {
      const response = await this.client.get("/api/health");
      return response.data;
    } catch (error) {
      console.error("Error en health check:", error);
      throw new Error("Sistema no disponible");
    }
  }

  /**
   * Obtiene metadatos de películas (géneros y años disponibles)
   * @returns {Promise<{genres: string[], years: number[]}>}
   */
  async getMovieMeta() {
    try {
      const response = await this.client.get("/api/v1/meta/movies");
      return response.data;
    } catch (error) {
      console.error("Error obteniendo metadatos de películas:", error);
      throw new Error(
        error.response?.data?.detail ||
          "Error obteniendo metadatos de películas"
      );
    }
  }

  /**
   * Obtiene metadatos de aerolíneas (orígenes, destinos, aerolíneas)
   * @returns {Promise<{origins: string[], destinations: string[], carriers: string[]}>}
   */
  async getAirlineMeta() {
    try {
      const response = await this.client.get("/api/v1/meta/airports");
      return response.data;
    } catch (error) {
      console.error("Error obteniendo metadatos de aeropuertos:", error);
      throw new Error(
        error.response?.data?.detail ||
          "Error obteniendo metadatos de aeropuertos"
      );
    }
  }
}

// Exportar instancia única del cliente
export const apiClient = new APIClient();
