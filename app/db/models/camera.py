from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), nullable=False)
    ip_address = Column(String(16), nullable=False)

    site_location_id = Column(
        Integer,
        ForeignKey("site_locations.id", ondelete="CASCADE"),
        nullable=False
    )

    # 1 = low light, 2 = high traffic
    location_type = Column(
        Integer,
        nullable=False,
        server_default="1"
    )

    is_active = Column(
        Boolean,
        nullable=False,
        server_default="true"
    )

    # bidirectional relationship
    site_location = relationship(
        "SiteLocation",
        back_populates="cameras"
    )

