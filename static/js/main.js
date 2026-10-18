document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const selectedFilesContainer = document.getElementById("selected-files-container");
    const fileCountSpan = document.getElementById("file-count");
    const fileListDiv = document.getElementById("file-list");
    const samplesGrid = document.getElementById("samples-grid");
    const analyzeBtn = document.getElementById("analyze-btn");
    const clearBtn = document.getElementById("clear-btn");
    const loader = document.getElementById("analysis-loader");
    const loaderProgress = document.getElementById("loader-progress");
    const resultsSection = document.getElementById("results-section");
    const resultsGrid = document.getElementById("results-grid");

    // Modal Elements
    const modal = document.getElementById("image-modal");
    const modalImage = document.getElementById("modal-image");
    const modalTitle = document.getElementById("modal-title");
    const modalMetricFire = document.getElementById("modal-metric-fire");
    const modalMetricHuman = document.getElementById("modal-metric-human");
    const modalMetricSmoke = document.getElementById("modal-metric-smoke");
    const modalPriorityBadge = document.getElementById("modal-priority-badge");
    const modalDispatchInstruction = document.getElementById("modal-dispatch-instruction");
    const modalCloseBtn = document.getElementById("modal-close-btn");
    const modalCloseBackdrop = document.getElementById("modal-close-backdrop");

    // State Variables
    let selectedUploadedFiles = [];
    let selectedSampleNames = new Set();
    let currentResults = [];

    // 1. Load Test Dataset Samples
    async function loadSamples() {
        try {
            const response = await fetch("/api/samples");
            const data = await response.json();

            if (data.samples && data.samples.length > 0) {
                samplesGrid.innerHTML = "";
                data.samples.forEach(sample => {
                    const card = document.createElement("div");
                    card.className = "sample-card";
                    card.dataset.name = sample;

                    const img = document.createElement("img");
                    img.src = `/static/samples/${sample}`;
                    img.alt = sample;
                    img.loading = "lazy";

                    const overlay = document.createElement("div");
                    overlay.className = "check-overlay";
                    overlay.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;

                    card.appendChild(img);
                    card.appendChild(overlay);

                    card.addEventListener("click", () => toggleSampleSelection(sample, card));
                    samplesGrid.appendChild(card);
                });
            } else {
                samplesGrid.innerHTML = `<p class="section-desc">No test samples found in the static/samples folder.</p>`;
            }
        } catch (error) {
            console.error("Failed to load sample list:", error);
            samplesGrid.innerHTML = `<p class="section-desc">Error loading sample dataset.</p>`;
        }
    }

    function toggleSampleSelection(name, cardElement) {
        if (selectedSampleNames.has(name)) {
            selectedSampleNames.delete(name);
            cardElement.classList.remove("selected");
        } else {
            selectedSampleNames.add(name);
            cardElement.classList.add("selected");
        }
        updateUIState();
    }

    // 2. Drag & Drop Upload Handlers
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFilesSelected(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFilesSelected(e.target.files);
        }
    });

    function handleFilesSelected(fileList) {
        for (let i = 0; i < fileList.length; i++) {
            const file = fileList[i];
            // Check if file is already added
            if (!selectedUploadedFiles.some(f => f.name === file.name && f.size === file.size)) {
                selectedUploadedFiles.push(file);
            }
        }
        renderSelectedFileList();
        updateUIState();
    }

    function renderSelectedFileList() {
        if (selectedUploadedFiles.length > 0) {
            selectedFilesContainer.style.display = "block";
            fileCountSpan.textContent = selectedUploadedFiles.length;
            fileListDiv.innerHTML = "";

            selectedUploadedFiles.forEach((file, index) => {
                const item = document.createElement("div");
                item.className = "file-item";

                const name = document.createElement("span");
                name.className = "file-item-name";
                name.textContent = file.name;

                const removeBtn = document.createElement("button");
                removeBtn.className = "file-remove-btn";
                removeBtn.textContent = "Remove";
                removeBtn.addEventListener("click", () => {
                    selectedUploadedFiles.splice(index, 1);
                    renderSelectedFileList();
                    updateUIState();
                });

                item.appendChild(name);
                item.appendChild(removeBtn);
                fileListDiv.appendChild(item);
            });
        } else {
            selectedFilesContainer.style.display = "none";
        }
    }

    // 3. UI Helper States
    function updateUIState() {
        const totalSelected = selectedUploadedFiles.length + selectedSampleNames.size;
        analyzeBtn.disabled = totalSelected < 3;

        if (totalSelected >= 3) {
            analyzeBtn.querySelector("span").textContent = `Analyze & Rank ${totalSelected} Scenes`;
        } else {
            analyzeBtn.querySelector("span").textContent = "Analyze & Rank Scenes";
        }
    }

    function clearAll() {
        selectedUploadedFiles = [];
        selectedSampleNames.clear();
        fileInput.value = "";
        renderSelectedFileList();

        // Remove selection borders on sample cards
        document.querySelectorAll(".sample-card").forEach(card => {
            card.classList.remove("selected");
        });

        resultsSection.style.display = "none";
        resultsGrid.innerHTML = "";
        updateUIState();
    }

    clearBtn.addEventListener("click", clearAll);

    // 4. API Request Handler
    analyzeBtn.addEventListener("click", async () => {
        const totalSelected = selectedUploadedFiles.length + selectedSampleNames.size;
        if (totalSelected < 3) return;

        // Show Loader & Reset progress fill
        loader.style.display = "flex";
        loaderProgress.style.width = "0%";

        // Simulate progress bar increments
        let progress = 0;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.floor(Math.random() * 15) + 5;
                if (progress > 90) progress = 90;
                loaderProgress.style.width = `${progress}%`;
            }
        }, 300);

        try {
            const formData = new FormData();

            // Append uploaded files
            selectedUploadedFiles.forEach(file => {
                formData.append("images", file);
            });

            // Append selected sample filenames
            selectedSampleNames.forEach(sampleName => {
                formData.append("samples", sampleName);
            });

            const response = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            clearInterval(progressInterval);
            loaderProgress.style.width = "100%";

            setTimeout(() => {
                loader.style.display = "none";
                if (response.ok) {
                    renderResults(data.results);
                } else {
                    alert(data.error || "An error occurred during analysis.");
                }
            }, 500);

        } catch (error) {
            clearInterval(progressInterval);
            loader.style.display = "none";
            console.error("Inference analysis failed:", error);
            alert("Connection error: Failed to reach the backend triage server.");
        }
    });

    // 5. Render Ranked Results Cards
    function renderResults(results) {
        currentResults = results;
        resultsGrid.innerHTML = "";
        resultsSection.style.display = "block";

        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth" });

        results.forEach((item, index) => {
            const card = document.createElement("div");
            card.className = `card glass-card incident-card p-${item.color_class}`;
            card.style.animationDelay = `${index * 150}ms`;

            // Card Header
            const cardHeader = document.createElement("div");
            cardHeader.className = "incident-rank-header";
            cardHeader.innerHTML = `
                <span class="rank-badge">Incident Rank #${index + 1}</span>
                <span class="badge ${getPriorityBadgeClass(item.color_class)}">${item.priority_badge}</span>
            `;

            // Image Container
            const imgWrapper = document.createElement("div");
            imgWrapper.className = "incident-image-wrapper";

            const img = document.createElement("img");
            img.src = item.annotated_url || "/static/samples/" + item.original_name;
            img.alt = item.original_name;

            const overlay = document.createElement("div");
            overlay.className = "view-overlay";
            overlay.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>`;

            imgWrapper.appendChild(img);
            imgWrapper.appendChild(overlay);
            imgWrapper.addEventListener("click", () => openLightbox(item, index));

            // Content Meta Area
            const metaDiv = document.createElement("div");
            metaDiv.className = "incident-meta";

            const title = document.createElement("h4");
            title.className = "incident-title";
            title.textContent = item.original_name;

            const metrics = document.createElement("div");
            metrics.className = "detection-metrics";
            metrics.innerHTML = `
                <span class="metric-tag">🔥 ${item.counts.fire} Fire</span>
                <span class="metric-tag">👤 ${item.counts.human} Humans</span>
                <span class="metric-tag">💨 ${item.counts.smoke} Smoke</span>
            `;

            const dispatchBlock = document.createElement("div");
            dispatchBlock.className = "dispatch-instruction-block";
            dispatchBlock.textContent = item.dispatch_info;

            metaDiv.appendChild(title);
            metaDiv.appendChild(metrics);
            metaDiv.appendChild(dispatchBlock);

            card.appendChild(cardHeader);
            card.appendChild(imgWrapper);
            card.appendChild(metaDiv);

            resultsGrid.appendChild(card);
        });
    }

    function getPriorityBadgeClass(colorClass) {
        if (colorClass === "danger" || colorClass === "danger-single") return "badge-danger";
        if (colorClass === "warning" || colorClass === "warning-single") return "badge-warning";
        return "badge-success";
    }

    // 6. Lightbox modal popup handlers
    function openLightbox(item, index) {
        modalImage.src = item.annotated_url || "/static/samples/" + item.original_name;
        modalTitle.textContent = `Incident Triage Report - #${index + 1}: ${item.original_name}`;

        modalMetricFire.textContent = `🔥 ${item.counts.fire} Fire`;
        modalMetricHuman.textContent = `👤 ${item.counts.human} Humans`;
        modalMetricSmoke.textContent = `💨 ${item.counts.smoke} Smoke`;

        // Remove prior class names and add current
        modalPriorityBadge.className = `badge ${getPriorityBadgeClass(item.color_class)}`;
        modalPriorityBadge.textContent = item.priority_badge;

        modalDispatchInstruction.textContent = item.dispatch_info;

        // Render border/shadow colors in modal priority container
        const priorityContainer = document.getElementById("modal-priority-container");
        priorityContainer.style.borderLeftColor = getHexForColorClass(item.color_class);

        modal.style.display = "flex";
        document.body.style.overflow = "hidden"; // Disable background scrolling
    }

    function getHexForColorClass(colorClass) {
        switch (colorClass) {
            case "danger": return "#ef4444";
            case "danger-single": return "#dc2626";
            case "warning": return "#f97316";
            case "warning-single": return "#ea580c";
            case "info": return "#eab308";
            case "success": return "#10b981";
            default: return "#6366f1";
        }
    }

    function closeLightbox() {
        modal.style.display = "none";
        document.body.style.overflow = "";
    }

    modalCloseBtn.addEventListener("click", closeLightbox);
    modalCloseBackdrop.addEventListener("click", closeLightbox);
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeLightbox();
    });

    // YouTube video analysis handlers
    const youtubeUrlInput = document.getElementById("youtube-url-input");
    const analyzeVideoBtn = document.getElementById("analyze-video-btn");
    const videoPreviewContainer = document.getElementById("video-preview-container");
    const youtubeIframe = document.getElementById("youtube-iframe");

    function getYouTubeEmbedUrl(url) {
        if (!url) return null;
        let videoId = null;
        const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
        const match = url.match(regExp);
        if (match && match[2].length === 11) {
            videoId = match[2];
        }
        return videoId ? `https://www.youtube.com/embed/${videoId}` : null;
    }

    function updateVideoPreview() {
        const url = youtubeUrlInput.value.trim();
        const embedUrl = getYouTubeEmbedUrl(url);
        if (embedUrl) {
            youtubeIframe.src = embedUrl;
            videoPreviewContainer.style.display = "block";
        } else {
            youtubeIframe.src = "";
            videoPreviewContainer.style.display = "none";
        }
    }

    // Update preview when URL changes
    youtubeUrlInput.addEventListener("input", updateVideoPreview);

    // Initial update for default URL
    updateVideoPreview();

    analyzeVideoBtn.addEventListener("click", async () => {
        const url = youtubeUrlInput.value.trim();
        if (!url) {
            alert("Please enter a valid YouTube URL first.");
            return;
        }

        // Show Loader & Reset progress fill
        loader.style.display = "flex";
        loaderProgress.style.width = "0%";

        // Show youtube specific loader text
        const loaderTitle = loader.querySelector("h3");
        const loaderDesc = loader.querySelector("p");
        const originalTitle = loaderTitle.textContent;
        const originalDesc = loaderDesc.textContent;

        loaderTitle.textContent = "Downloading & Processing YouTube POV Video";
        loaderDesc.textContent = "Connecting to streaming server, extracting key frames, and analyzing incidents...";

        // Simulate progress bar increments
        let progress = 0;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.floor(Math.random() * 8) + 3; // slower progress since it takes longer
                if (progress > 90) progress = 90;
                loaderProgress.style.width = `${progress}%`;
            }
        }, 500);

        try {
            const response = await fetch("/api/analyze-video", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ url: url })
            });

            const data = await response.json();

            clearInterval(progressInterval);
            loaderProgress.style.width = "100%";

            setTimeout(() => {
                loader.style.display = "none";
                // Reset loader text
                loaderTitle.textContent = originalTitle;
                loaderDesc.textContent = originalDesc;

                if (response.ok) {
                    // Update results section header with video title if available
                    const resultsSummary = document.getElementById("results-summary");
                    if (resultsSummary && data.title) {
                        resultsSummary.textContent = `Sorted: Highest Priority First | Video: ${data.title}`;
                    }
                    renderResults(data.results);
                } else {
                    alert(data.error || "An error occurred during video analysis.");
                }
            }, 500);

        } catch (error) {
            clearInterval(progressInterval);
            loader.style.display = "none";
            loaderTitle.textContent = originalTitle;
            loaderDesc.textContent = originalDesc;
            console.error("Video analysis failed:", error);
            alert("Connection error: Failed to reach the backend triage server.");
        }
    });

    // --- Multi-Room Triage Frontend Logic ---
    const navDashboard = document.getElementById("nav-dashboard");
    const navMultiRoom = document.getElementById("nav-multi-room");
    const navQuickTest = document.getElementById("nav-quick-test");

    const uploadTriageView = document.getElementById("upload-triage-view");
    const multiRoomView = document.getElementById("multi-room-view");

    function switchTab(activeNav, showUpload) {
        [navDashboard, navMultiRoom, navQuickTest].forEach(nav => {
            if (nav) nav.classList.remove("active");
        });
        if (activeNav) activeNav.classList.add("active");

        if (showUpload) {
            uploadTriageView.style.display = "block";
            multiRoomView.style.display = "none";
        } else {
            uploadTriageView.style.display = "none";
            multiRoomView.style.display = "block";
        }
    }

    if (navDashboard) {
        navDashboard.addEventListener("click", (e) => {
            e.preventDefault();
            switchTab(navDashboard, true);
        });
    }
    if (navMultiRoom) {
        navMultiRoom.addEventListener("click", (e) => {
            e.preventDefault();
            switchTab(navMultiRoom, false);
        });
    }
    if (navQuickTest) {
        navQuickTest.addEventListener("click", (e) => {
            // Keep dashboard view active while scrolling to sample image gallery anchor
            switchTab(navDashboard, true);
        });
    }

    const runMultiTriageBtn = document.getElementById("run-multi-triage-btn");
    const multiRoomResults = document.getElementById("multi-room-results");
    const roomFeedsGrid = document.getElementById("room-feeds-grid");
    const sarRankedList = document.getElementById("sar-ranked-list");

    if (runMultiTriageBtn) {
        runMultiTriageBtn.addEventListener("click", async () => {
            loader.style.display = "flex";
            loaderProgress.style.width = "0%";

            const loaderTitle = loader.querySelector("h3");
            const loaderDesc = loader.querySelector("p");
            const originalTitle = loaderTitle.textContent;
            const originalDesc = loaderDesc.textContent;

            loaderTitle.textContent = "Analyzing Rooms & Calculating SAR Priority";
            loaderDesc.textContent = "Downloading live streams, running YOLO11s-t1 inference on key room frames, and ranking rescue urgency...";

            let progress = 0;
            const progressInterval = setInterval(() => {
                if (progress < 90) {
                    progress += Math.floor(Math.random() * 4) + 1;
                    if (progress > 90) progress = 90;
                    loaderProgress.style.width = `${progress}%`;
                }
            }, 800);

            try {
                const response = await fetch("/api/analyze-multi-room", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    }
                });

                const data = await response.json();

                clearInterval(progressInterval);
                loaderProgress.style.width = "100%";

                setTimeout(() => {
                    loader.style.display = "none";
                    loaderTitle.textContent = originalTitle;
                    loaderDesc.textContent = originalDesc;

                    if (response.ok) {
                        renderMultiRoomResults(data.results, data.ranked_results);
                    } else {
                        alert(data.error || "An error occurred during multi-room triage.");
                    }
                }, 500);
            } catch (error) {
                clearInterval(progressInterval);
                loader.style.display = "none";
                loaderTitle.textContent = originalTitle;
                loaderDesc.textContent = originalDesc;
                console.error("Multi-room analysis failed:", error);
                alert("Connection error: Failed to reach the backend triage server.");
            }
        });
    }

    function renderMultiRoomResults(results, rankedResults) {
        multiRoomResults.style.display = "block";
        multiRoomResults.scrollIntoView({ behavior: "smooth" });

        // 1. Render Part 1: Room Feeds Grid
        roomFeedsGrid.innerHTML = "";
        results.forEach((item, index) => {
            const card = document.createElement("div");
            card.className = `card glass-card incident-card p-${item.color_class}`;
            card.style.animationDelay = `${index * 100}ms`;

            card.innerHTML = `
                <div class="incident-rank-header">
                    <span class="rank-badge" style="color:var(--accent); font-weight:700;">${item.room_id}</span>
                    <span class="badge ${getPriorityBadgeClass(item.color_class)}">${item.priority_badge}</span>
                </div>
                <div class="incident-image-wrapper">
                    <img src="${item.annotated_url || '/static/samples/image_894_jpg.rf.6b4758f5e0938d6fc55e911d3331f6ad.jpg'}" alt="${item.room_id}">
                    <div class="view-overlay">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
                    </div>
                </div>
                <div class="incident-meta">
                    <h4 class="incident-title" style="font-size:0.85rem; height:40px; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;">${item.video_title}</h4>
                    <div class="detection-metrics" style="margin-top:6px;">
                        <span class="metric-tag">🔥 ${item.counts.fire} Fire</span>
                        <span class="metric-tag">👤 ${item.counts.human} Humans</span>
                        <span class="metric-tag">💨 ${item.counts.smoke} Smoke</span>
                    </div>
                    <div class="dispatch-instruction-block" style="margin-top:10px; font-size:0.8rem;">
                        ${item.dispatch_info}
                    </div>
                </div>
            `;

            const imgWrapper = card.querySelector(".incident-image-wrapper");
            imgWrapper.addEventListener("click", () => {
                openLightbox({
                    original_name: item.room_id + " - " + item.video_title,
                    annotated_url: item.annotated_url,
                    counts: item.counts,
                    color_class: item.color_class,
                    priority_badge: item.priority_badge,
                    dispatch_info: item.dispatch_info
                }, index);
            });

            roomFeedsGrid.appendChild(card);
        });

        // 2. Render Part 2: Ranked timeline
        sarRankedList.innerHTML = "";
        rankedResults.forEach((item, index) => {
            const timelineCard = document.createElement("div");
            timelineCard.className = `sar-timeline-card p-${item.color_class}`;
            timelineCard.style.animationDelay = `${index * 150}ms`;

            timelineCard.innerHTML = `
                <div class="sar-timeline-rank">
                    <span class="sar-rank-num">#${index + 1}</span>
                    <span class="sar-rank-lbl">SAR Rank</span>
                </div>
                <div class="sar-timeline-img-wrapper">
                    <img src="${item.annotated_url || '/static/samples/image_894_jpg.rf.6b4758f5e0938d6fc55e911d3331f6ad.jpg'}" alt="${item.room_id}">
                </div>
                <div class="sar-timeline-content">
                    <div class="sar-timeline-header">
                        <span class="sar-timeline-title" style="font-weight: 700; color: white;">${item.room_id} (${item.video_title})</span>
                        <span class="badge ${getPriorityBadgeClass(item.color_class)}">${item.priority_badge}</span>
                    </div>
                    <div class="sar-timeline-metrics" style="margin: 6px 0;">
                        <span class="metric-tag">🔥 ${item.counts.fire} Fire</span>
                        <span class="metric-tag">👤 ${item.counts.human} Humans</span>
                        <span class="metric-tag">💨 ${item.counts.smoke} Smoke</span>
                    </div>
                    <div style="font-size:0.85rem; color:var(--text-secondary);">
                        <strong>Dispatch Instructions:</strong> ${item.dispatch_info}
                    </div>
                </div>
            `;

            const imgWrapper = timelineCard.querySelector(".sar-timeline-img-wrapper");
            imgWrapper.addEventListener("click", () => {
                openLightbox({
                    original_name: item.room_id + " - " + item.video_title,
                    annotated_url: item.annotated_url,
                    counts: item.counts,
                    color_class: item.color_class,
                    priority_badge: item.priority_badge,
                    dispatch_info: item.dispatch_info
                }, index);
            });

            sarRankedList.appendChild(timelineCard);
        });
    }

    // Initialize Page
    loadSamples();
});
