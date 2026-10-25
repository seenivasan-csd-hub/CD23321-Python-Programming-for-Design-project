(() => {
  document.querySelectorAll("[data-password-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const input = document.getElementById(button.dataset.passwordToggle);
      if (!input) return;
      const showing = input.type === "password";
      input.type = showing ? "text" : "password";
      button.textContent = showing ? "Hide" : "Show";
      button.setAttribute("aria-pressed", String(showing));
      button.setAttribute("aria-label", `${showing ? "Hide" : "Show"} password`);
    });
  });

  const setStatus = (form, selector, message) => {
    const target = form.querySelector(selector);
    if (target) target.textContent = message;
  };

  document.querySelectorAll("[data-geolocate]").forEach((button) => {
    button.addEventListener("click", () => {
      const form = button.closest("form");
      if (!navigator.geolocation) {
        setStatus(form, "[data-location-status]", "This browser does not provide location access. Enter coordinates manually.");
        return;
      }
      setStatus(form, "[data-location-status]", "Waiting for browser location permission…");
      navigator.geolocation.getCurrentPosition((position) => {
        const latitude = form.querySelector('[name="latitude"]');
        const longitude = form.querySelector('[name="longitude"]');
        latitude.value = position.coords.latitude.toFixed(6);
        longitude.value = position.coords.longitude.toFixed(6);
        setStatus(form, "[data-location-status]", "Coordinates filled in. They will be saved only if you submit the farm profile.");
      }, (error) => {
        const message = error.code === error.PERMISSION_DENIED ? "Location permission was denied. You can enter coordinates manually." : error.code === error.TIMEOUT ? "Location lookup timed out. Try again or enter coordinates manually." : "Location is unavailable. Enter coordinates manually.";
        setStatus(form, "[data-location-status]", message);
      }, { enableHighAccuracy: false, timeout: 12000, maximumAge: 300000 });
    });
  });

  document.querySelectorAll("form[data-weather-form]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      const submit = event.submitter || form.querySelector("[data-weather-submit]");
      form.setAttribute("aria-busy", "true");
      setStatus(form, "[data-location-status]", "Loading current weather and forecast from AgroMonitoring...");
      if (submit) {
        submit.disabled = true;
        submit.textContent = "Loading weather...";
      }
    });
  });

  document.querySelectorAll("[data-camera-start]").forEach((startButton) => {
    const form = startButton.closest("form");
    const video = form.querySelector("[data-camera-preview]");
    const captureButton = form.querySelector("[data-camera-capture]");
    const fileInput = form.querySelector('input[type="file"][name="leaf_image"]');
    let stream;

    startButton.addEventListener("click", async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus(form, "[data-camera-status]", "Camera access is not available here. Choose an image file instead.");
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
        video.srcObject = stream;
        video.hidden = false;
        captureButton.disabled = false;
        setStatus(form, "[data-camera-status]", "Camera is on. Capture a close, well-lit leaf photo when ready.");
      } catch (error) {
        const message = error.name === "NotAllowedError" ? "Camera permission was denied. You can upload an image instead." : "Camera could not be opened. You can upload an image instead.";
        setStatus(form, "[data-camera-status]", message);
      }
    });

    captureButton.addEventListener("click", () => {
      if (!video.videoWidth || !video.videoHeight) {
        setStatus(form, "[data-camera-status]", "Camera is not ready yet. Wait a moment and try again.");
        return;
      }
      const canvas = form.querySelector("[data-camera-canvas]");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      canvas.toBlob((blob) => {
        if (!blob) {
          setStatus(form, "[data-camera-status]", "Photo capture failed. Try again or upload an image.");
          return;
        }
        const transfer = new DataTransfer();
        transfer.items.add(new File([blob], "leaf-capture.jpg", { type: "image/jpeg" }));
        fileInput.files = transfer.files;
        fileInput.dispatchEvent(new Event("change", { bubbles: true }));
        setStatus(form, "[data-camera-status]", "Photo captured and ready to scan. The camera has been turned off.");
        if (stream) stream.getTracks().forEach((track) => track.stop());
        video.srcObject = null;
        video.hidden = true;
        captureButton.disabled = true;
      }, "image/jpeg", 0.9);
    });
  });

  document.querySelectorAll("form[data-scan-form]").forEach((form) => {
    const input = form.querySelector("[data-image-input]");
    const preview = form.querySelector("[data-image-preview]");
    const progress = form.querySelector("[data-scan-progress]");
    const status = form.querySelector("[data-scan-status]");
    const submit = form.querySelector("[data-scan-submit]");
    let previewUrl;

    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = file ? URL.createObjectURL(file) : null;
      preview.src = previewUrl || "";
      preview.hidden = !previewUrl;
    });

    form.addEventListener("submit", () => {
      progress.hidden = false;
      form.setAttribute("aria-busy", "true");
      status.textContent = "Scanning image locally. This can take a few seconds.";
      submit.disabled = true;
      submit.textContent = "Scanning…";
    });
  });

  const helpLocation = document.querySelector("[data-help-location-input]");
  const helpMap = document.querySelector("[data-help-map]");
  const helpStatus = document.querySelector("[data-help-location-status]");
  const helpGeolocate = document.querySelector("[data-help-geolocate]");
  if (helpLocation && helpMap && helpStatus && helpGeolocate) {
    const updateHelpMap = () => {
      const place = helpLocation.value.trim();
      if (!place) {
        helpMap.hidden = true;
        helpMap.removeAttribute("href");
        return false;
      }
      const query = `Krishi Vigyan Kendra agricultural extension plant disease diagnostic service near ${place}`;
      helpMap.href = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(query)}`;
      helpMap.hidden = false;
      return true;
    };

    helpLocation.addEventListener("input", updateHelpMap);
    helpGeolocate.addEventListener("click", () => {
      if (!navigator.geolocation) {
        helpStatus.textContent = "This browser does not provide location access. Enter a village, town, district or PIN code.";
        return;
      }
      helpStatus.textContent = "Waiting for your browser location permission…";
      navigator.geolocation.getCurrentPosition((position) => {
        helpLocation.value = `${position.coords.latitude.toFixed(6)}, ${position.coords.longitude.toFixed(6)}`;
        updateHelpMap();
        helpStatus.textContent = "Location added to the Google Maps search link. FarmIQ did not store or send it to its server.";
      }, (error) => {
        helpStatus.textContent = error.code === error.PERMISSION_DENIED
          ? "Location permission was denied. Enter a village, town, district or PIN code instead."
          : error.code === error.TIMEOUT
            ? "Location lookup timed out. Enter a village, town, district or PIN code instead."
            : "Location is unavailable. Enter a village, town, district or PIN code instead.";
      }, { enableHighAccuracy: false, timeout: 12000, maximumAge: 300000 });
    });
  }

  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!window.confirm(form.dataset.confirm)) event.preventDefault();
    });
  });
})();
