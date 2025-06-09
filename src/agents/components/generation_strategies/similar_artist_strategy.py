from typing import Dict, Any, List

class SimilarArtistStrategy:
    async def generate_candidates(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        max_candidates: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate enhanced similar artist candidates using multi-layered approach.
        
        Uses SimilarityExplorer for deep discovery and sophisticated filtering.
        """
        try:
            # Extract target artist
            target_artists = entities.get('artists', [])
            if not target_artists:
                self.logger.warning("No target artists found for similarity search")
                return []
            
            target_artist = target_artists[0]
            self.logger.info(f"ENHANCED SIMILAR ARTIST GENERATION: Finding artists similar to {target_artist}")
            
            # Use SimilarityExplorer for multi-hop discovery
            all_candidates = []
            
            # Phase 1: Direct similarity exploration
            similar_tracks = await self.similarity_explorer.explore_similar_tracks(
                target_artist, max_tracks=max_candidates * 2
            )
            
            # 🔍 DEBUG: Log similarity explorer results
            self.logger.info(f"🔍 SIMILARITY EXPLORER: Found {len(similar_tracks)} tracks")
            if similar_tracks:
                sample_tracks = similar_tracks[:3]
                for i, track in enumerate(sample_tracks):
                    listeners = getattr(track, 'listeners', 0)
                    playcount = getattr(track, 'playcount', 0)
                    self.logger.info(
                        f"🔍 Explorer {i+1}: {getattr(track, 'artist', 'Unknown')} - {getattr(track, 'name', 'Unknown')} "
                        f"(listeners: {listeners}, playcount: {playcount})"
                    )
            
            # Convert to candidate format with enhanced metadata
            for track in similar_tracks:
                candidate = await self._enhance_similarity_track(track, target_artist)
                
                # 🚨 CRITICAL: Validate candidate has real data
                listeners = candidate.get('listeners', 0)
                playcount = candidate.get('playcount', 0)
                
                if listeners == 0 and playcount == 0:
                    self.logger.warning(
                        f"🚨 SKIPPING INVALID CANDIDATE: {candidate.get('artist', 'Unknown')} - {candidate.get('name', 'Unknown')} "
                        f"(listeners: {listeners}, playcount: {playcount}) - No popularity data"
                    )
                    continue  # Skip candidates with no real data
                
                all_candidates.append(candidate)
            
            # Phase 2: Apply style-aware filtering
            filtered_candidates = await self._apply_style_similarity_filter(
                all_candidates, target_artist, max_candidates
            )
            
            # 🔍 DEBUG: Log final candidates
            self.logger.info(f"🔍 ENHANCED STRATEGY OUTPUT: {len(filtered_candidates)} high-quality candidates")
            if filtered_candidates:
                sample_final = filtered_candidates[:5]
                for i, candidate in enumerate(sample_final):
                    self.logger.info(
                        f"🔍 Enhanced {i+1}: {candidate.get('artist', 'Unknown')} - {candidate.get('name', 'Unknown')} "
                        f"(listeners: {candidate.get('listeners', 0)}, similarity: {candidate.get('similarity_score', 0):.3f})"
                    )
            
            self.logger.info(f"TOTAL ENHANCED SIMILARITY CANDIDATES: {len(filtered_candidates)} tracks")
            return filtered_candidates
            
        except Exception as e:
            self.logger.error(f"Enhanced similar artist generation failed: {e}")
            return [] 