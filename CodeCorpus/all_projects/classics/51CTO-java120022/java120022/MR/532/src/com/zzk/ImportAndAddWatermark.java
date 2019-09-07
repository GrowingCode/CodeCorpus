package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Image;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfReader;
import com.itextpdf.text.pdf.PdfStamper;
import com.itextpdf.text.pdf.PdfWriter;

public class ImportAndAddWatermark {
	public static void main(String[] args) {
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("c:\\水印.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			document.add(new Paragraph("No. One.")); // 向文档中添加内容
			document.add(new Paragraph("No. Two.")); // 向文档中添加内容
			document.add(new Paragraph("No. Three.")); // 向文档中添加内容
			document.add(new Paragraph("No. Four.")); // 向文档中添加内容
			document.add(new Paragraph("No. Five.")); // 向文档中添加内容
			document.close();// 关闭文档对象
			PdfReader reader = new PdfReader("c:\\水印.pdf");// 创建“水印.pdf”的PdfReader对象
			PdfStamper stamp = new PdfStamper(reader, new FileOutputStream("c:\\导入并添加水印.pdf"));// 创建PdfStamper对象
			Image img = Image.getInstance("image/watermark.jpg");// 写上内容
			img.setAbsolutePosition(30, 385);// 定位图片对象
			PdfContentByte under = stamp.getUnderContent(1);// 获得第一页的内容
			under.addImage(img);// 添加图片,完成水印功能
			stamp.close();// PdfStamper对象，将从“水印.pdf”中读取的文档添加水印后写入“导入并添加水印.pdf”
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
